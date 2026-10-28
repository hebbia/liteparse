//! Emission-layer reconciliation.
//!
//! Structured emission (tables, paragraphs, figures) sometimes drops text
//! runs that extraction produced. This module audits each page's rendered
//! markdown against its extracted `text_items` and re-emits every run no
//! block accounted for — spliced back into the flow at its reading
//! position (runs whose centre falls inside a table's box are grouped
//! right after that table under a `liteparse:reconciled-table` marker, as
//! a signal the grid itself lost cells).
//!
//! Detection is per-run contiguous coverage, not bag-of-tokens: each run's
//! text is normalized to a lowercase alphanumeric stream and checked via
//! sliding 8-byte shingles against the emitted page's stream. A run counts
//! as emitted only where its own character sequence survives — common
//! words elsewhere in the document cannot mask a dropped run.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};

use crate::config::TableFormat;
use crate::layout::LayoutBlock;
use crate::markdown_layout::PositionedBlock;
use crate::types::ParsedPage;
use crate::types::Rect;

const TABLE_KINDS: [&str; 3] = ["table", "merged_table", "grid_fallback"];
/// Sliding-window size over the normalized alphanumeric stream.
const SHINGLE: usize = 8;
/// A run counts as emitted only when at least this fraction of its
/// shingles survive in the rendered page.
const MIN_MISSING_FRACTION: f32 = 0.5;

/// Strip markdown scaffolding that never carries payload text, so it does
/// not count toward emitted coverage: HTML tags, image/link targets
/// (`](…)` and `![alt](…)`), HTML comments.
fn strip_scaffolding(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut chars = markdown.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '<' => {
                // Drop through the closing '>' (or end of string).
                for c in chars.by_ref() {
                    if c == '>' {
                        break;
                    }
                }
            }
            ']' => {
                // A link/image target follows: `](…)` — drop the parens.
                if chars.peek() == Some(&'(') {
                    chars.next();
                    let mut depth = 1;
                    for c in chars.by_ref() {
                        match c {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                } else {
                    out.push(c);
                }
            }
            _ => out.push(c),
        }
    }
    out
}

fn alnum_stream(s: &str) -> String {
    s.chars()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Hash of one byte shingle over the normalized stream, using the shared
/// `RandomState` so every hash in a pass is consistent. Bytes rather than
/// `&str` slices because non-ASCII alphanumerics are multi-byte — a fixed
/// byte window can split a UTF-8 sequence, which is fine for hashing but
/// would panic as a `&str` slice.
fn shingle_hash(rs: &std::collections::hash_map::RandomState, b: &[u8]) -> u64 {
    let mut h = rs.build_hasher();
    b.hash(&mut h);
    h.finish()
}

fn shingle_hashes(
    rs: &std::collections::hash_map::RandomState,
    stream: &str,
    out: &mut Vec<u64>,
) {
    let b = stream.as_bytes();
    if b.len() < SHINGLE {
        return;
    }
    for i in 0..=b.len() - SHINGLE {
        out.push(shingle_hash(rs, &b[i..i + SHINGLE]));
    }
}

fn point_in(r: &Rect, x: f32, y: f32) -> bool {
    x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height
}

struct DroppedRun<'a> {
    x: f32,
    y: f32,
    in_table: bool,
    text: &'a str,
}

/// Every extracted run the rendered `markdown` failed to account for, in
/// viewport reading order (y then x). `in_table` marks runs whose centre
/// sits inside a detected table's box.
fn dropped_runs<'a>(
    page: &'a ParsedPage,
    markdown: &str,
    layout_blocks: &[LayoutBlock],
) -> Vec<DroppedRun<'a>> {
    let md_stream = alnum_stream(&strip_scaffolding(markdown));
    let rs = std::collections::hash_map::RandomState::new();
    let mut pool: HashMap<u64, u32> = HashMap::new();
    {
        let b = md_stream.as_bytes();
        if b.len() >= SHINGLE {
            for i in 0..=b.len() - SHINGLE {
                *pool.entry(shingle_hash(&rs, &b[i..i + SHINGLE])).or_insert(0) += 1;
            }
        }
    }

    let table_rects: Vec<Rect> = layout_blocks
        .iter()
        .filter(|b| TABLE_KINDS.contains(&b.kind.as_str()))
        .filter_map(|b| b.bbox.clone())
        .collect();

    // Reading order: viewport y then x. text_items keep extraction order,
    // which is content-stream order — right for layout inference, wrong for
    // inserting text a human should read.
    let mut items: Vec<&crate::types::TextItem> = page.text_items.iter().collect();
    items.sort_by(|a, b| a.y.total_cmp(&b.y).then_with(|| a.x.total_cmp(&b.x)));

    let mut dropped: Vec<DroppedRun> = Vec::new();
    for item in items {
        let stream = alnum_stream(&item.text);
        if stream.is_empty() {
            continue;
        }
        let is_dropped = if stream.len() < SHINGLE {
            // Too short to shingle: count as emitted iff the sequence
            // literally survives anywhere in the emitted stream.
            !md_stream.contains(&stream)
        } else {
            let mut item_shingles: Vec<u64> = Vec::new();
            shingle_hashes(&rs, &stream, &mut item_shingles);
            let mut missing = 0usize;
            for sh in &item_shingles {
                match pool.get_mut(sh) {
                    Some(n) if *n > 0 => *n -= 1,
                    _ => missing += 1,
                }
            }
            (missing as f32) > (item_shingles.len() as f32) * MIN_MISSING_FRACTION
        };
        if is_dropped {
            let cx = item.x + item.width / 2.0;
            let cy = item.y + item.height / 2.0;
            dropped.push(DroppedRun {
                x: cx,
                y: cy,
                in_table: table_rects.iter().any(|r| point_in(r, cx, cy)),
                text: item.text.trim(),
            });
        }
    }
    dropped
}

/// Index of the block a run should be emitted before: the first block
/// (in markdown/emission order) whose box starts clearly below the run's
/// centre, or end-of-page when none does. Blocks without a box carry no
/// position and are skipped.
fn flow_index(run: &DroppedRun, layout_blocks: &[LayoutBlock]) -> usize {
    for (i, b) in layout_blocks.iter().enumerate() {
        if let Some(r) = &b.bbox {
            if r.y > run.y {
                return i;
            }
        }
    }
    layout_blocks.len()
}

/// Index just after the deepest (smallest-area) table block containing the
/// run's centre, so interior drops land right next to the grid that lost
/// them. `None` when no table contains the run.
fn table_index(run: &DroppedRun, layout_blocks: &[LayoutBlock]) -> Option<usize> {
    layout_blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| TABLE_KINDS.contains(&b.kind.as_str()))
        .filter(|(_, b)| {
            b.bbox
                .as_ref()
                .map(|r| point_in(r, run.x, run.y))
                .unwrap_or(false)
        })
        .min_by(|(_, a), (_, b)| {
            let area = |lb: &LayoutBlock| {
                let r = lb.bbox.as_ref().unwrap();
                r.width * r.height
            };
            area(a)
                .partial_cmp(&area(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i + 1)
}

/// Write one insertion group (no leading/trailing separators — the caller
/// places them). Table-interior groups get a `liteparse:reconciled-table`
/// marker line.
fn push_group(out: &mut String, group: &mut Vec<&DroppedRun>, table: bool) {
    if group.is_empty() {
        return;
    }
    group.sort_by(|a, b| a.y.total_cmp(&b.y).then_with(|| a.x.total_cmp(&b.x)));
    if table {
        out.push_str("<!-- liteparse:reconciled-table -->\n");
    }
    for (i, r) in group.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(r.text);
    }
    group.clear();
}

/// Render `page`'s markdown and splice every extracted run the structured
/// emission dropped back into the flow at its reading position. When
/// `blocks` is `None` (page never classified) the projection text is
/// fenced as usual and dropped runs are appended after it — a fenced page
/// has no block geometry to splice into.
pub fn reconciled_page_markdown(
    page: &ParsedPage,
    blocks: Option<&[PositionedBlock]>,
    table_format: TableFormat,
) -> String {
    let base = crate::stages::render_page_markdown(page, blocks, table_format);
    let Some(bs) = blocks else {
        return append_dropped(page, &base);
    };
    if page.text_items.is_empty() {
        return base;
    }
    let layout_blocks = crate::layout::blocks_for_page(page, bs);
    let dropped = dropped_runs(page, &base, &layout_blocks);
    if dropped.is_empty() {
        return base;
    }

    // Assign each dropped run an emission index: before block `idx`
    // (idx == len means after the last block). Table-interior runs go right
    // after the table that should have contained them.
    let mut inserts: Vec<Vec<&DroppedRun>> = vec![Vec::new(); bs.len() + 1];
    let mut has_table_group = vec![false; bs.len() + 1];
    for r in &dropped {
        let idx = if r.in_table {
            match table_index(r, &layout_blocks) {
                Some(i) => {
                    has_table_group[i] = true;
                    i
                }
                None => flow_index(r, &layout_blocks),
            }
        } else {
            flow_index(r, &layout_blocks)
        };
        inserts[idx].push(r);
    }

    let mut out = String::with_capacity(base.len());
    for (i, pb) in bs.iter().enumerate() {
        if i > 0 {
            // Consecutive list items render as a tight list (single
            // newline) — but only when nothing is spliced between them.
            let tight = inserts[i].is_empty()
                && matches!(pb.block, crate::markdown_layout::Block::ListItem { .. })
                && matches!(
                    bs[i - 1].block,
                    crate::markdown_layout::Block::ListItem { .. }
                );
            out.push_str(if tight { "\n" } else { "\n\n" });
        }
        if !inserts[i].is_empty() {
            push_group(&mut out, &mut inserts[i], has_table_group[i]);
            out.push_str("\n\n");
        }
        out.push_str(&crate::markdown_layout::render_blocks_with(
            std::slice::from_ref(pb),
            table_format,
        ));
    }
    if !inserts[bs.len()].is_empty() {
        out.push_str("\n\n");
        push_group(&mut out, &mut inserts[bs.len()], has_table_group[bs.len()]);
    }
    out
}

/// Fence path (`blocks` is `None`): append dropped runs after the fenced
/// projection text, non-table lines first under `liteparse:reconciled`,
/// then any table-adjacent lines under `liteparse:reconciled-table`.
fn append_dropped(page: &ParsedPage, markdown: &str) -> String {
    let mut out = String::from(markdown);
    let dropped = dropped_runs(page, markdown, &[]);
    if dropped.is_empty() {
        return out;
    }
    out.push_str("\n\n<!-- liteparse:reconciled -->\n");
    let mut table_buf = String::new();
    for r in &dropped {
        if r.in_table {
            table_buf.push_str(r.text);
            table_buf.push('\n');
        } else {
            out.push_str(r.text);
            out.push('\n');
        }
    }
    if !table_buf.is_empty() {
        out.push_str("<!-- liteparse:reconciled-table -->\n");
        out.push_str(&table_buf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffolding_stripped() {
        let md = "<table><tr><td>a</td></tr></table>\n![](img_1.png)\n[x](http://y) b";
        let stream = alnum_stream(&strip_scaffolding(md));
        assert_eq!(stream, "axb");
    }

    #[test]
    fn stream_lowercase_alnum_only() {
        assert_eq!(alnum_stream("Héllo, Wörld! 42"), "héllowörld42");
    }

    #[test]
    fn shingle_windows() {
        let mut v: Vec<u64> = Vec::new();
        let rs = std::collections::hash_map::RandomState::new();
        shingle_hashes(&rs, "abcdefghij", &mut v);
        assert_eq!(v.len(), 3);
        assert_ne!(v[0], v[1]);
    }
}
