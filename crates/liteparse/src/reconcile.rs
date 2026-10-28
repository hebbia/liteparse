//! Emission reconcile pass.
//!
//! The markdown emitter decomposes projected lines into blocks (headings,
//! paragraphs, tables, figures). When classification or a table grid goes
//! wrong, the text it held can vanish from the rendered page even though
//! extraction found it. This pass audits each rendered page against the
//! page's extracted [`TextItem`] runs: any run whose text is not accounted
//! for by the emitted markdown is appended after the structured output, in
//! reading order. Runs sitting inside a table's bounding box are grouped
//! under a `liteparse:reconciled-table` marker so a downstream consumer can
//! tell plain-text fallbacks from cells a grid failed to claim.
//!
//! Coverage is measured on normalized character streams (lowercase
//! alphanumerics only, whitespace and markup removed): a run counts as
//! emitted only when its own 8-byte shingles appear in the rendered
//! markdown. A token-multiset comparison would credit common words that
//! appear anywhere on the page and miss real drops; contiguous shingles
//! only match where the run's text actually survived. Markdown scaffolding
//! (HTML tags, link targets, image refs) is stripped first so a `<td>` tag
//! never "covers" a dropped cell's text.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};

use crate::layout::LayoutBlock;
use crate::markdown_layout::PositionedBlock;
use crate::types::{ParsedPage, Rect};

/// Kinds whose bounding boxes mark a dropped run as table-adjacent.
const TABLE_KINDS: [&str; 3] = ["table", "merged_table", "grid_fallback"];

/// Shingle length for the emitted/missing comparison. Short enough to ride
/// out a one-character difference at a run's edge, long enough that matches
/// across unrelated text are rare.
const SHINGLE: usize = 8;

/// A run is re-emitted when more than half of its shingles have no
/// counterpart left in the emitted-markdown multiset.
const MIN_MISSING_FRACTION: f32 = 0.5;

/// Markdown scaffolding stripped before streaming: HTML tags (`<table>`,
/// `<td>`) and inline link/image targets (`](url)`, `![alt](url)`) would
/// otherwise donate text that masks real drops.
fn strip_scaffolding(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut chars = markdown.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '<' => {
                for c in chars.by_ref() {
                    if c == '>' {
                        break;
                    }
                }
            }
            ']' => {
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

/// Normalized character stream: lowercase alphanumeric characters only.
/// Whitespace, punctuation and markup all collapse away, so a run's
/// shingles match its rendered form even when the emitter rewrapped lines,
/// dropped soft hyphens, or reflowed a cell.
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

fn shingle_hashes(rs: &std::collections::hash_map::RandomState, stream: &str, out: &mut Vec<u64>) {
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

/// Audit `page`'s emitted `markdown` against its extracted text runs.
/// Returns the reconcile section to append (empty when nothing was dropped).
pub fn reconcile_page_markdown(
    page: &ParsedPage,
    markdown: &str,
    blocks: Option<&[PositionedBlock]>,
) -> String {
    if page.text_items.is_empty() {
        return String::new();
    }

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

    let table_rects: Vec<Rect> = blocks
        .map(|bs| crate::layout::blocks_for_page(page, bs))
        .unwrap_or_default()
        .iter()
        .filter(|b| TABLE_KINDS.contains(&b.kind.as_str()))
        .filter_map(|b: &LayoutBlock| b.bbox.clone())
        .collect();

    // Reading order: viewport y then x. text_items keep extraction order,
    // which is content-stream order — right for layout inference, wrong for
    // appending text a human should read.
    let mut items: Vec<&crate::types::TextItem> = page.text_items.iter().collect();
    items.sort_by(|a, b| a.y.total_cmp(&b.y).then_with(|| a.x.total_cmp(&b.x)));

    let mut dropped: Vec<(bool, &str)> = Vec::new();
    for item in items {
        let stream = alnum_stream(&item.text);
        if stream.is_empty() {
            continue;
        }
        let missing = if stream.len() < SHINGLE {
            // Too short to shingle: count as emitted iff the sequence
            // literally survives anywhere in the emitted stream.
            usize::from(!md_stream.contains(&stream))
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
            if (missing as f32) > (item_shingles.len() as f32) * MIN_MISSING_FRACTION {
                usize::MAX
            } else {
                0
            }
        };
        if missing > 0 {
            let cx = item.x + item.width / 2.0;
            let cy = item.y + item.height / 2.0;
            let in_table = table_rects.iter().any(|r| point_in(r, cx, cy));
            dropped.push((in_table, item.text.trim()));
        }
    }
    if dropped.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n\n<!-- liteparse:reconciled -->\n");
    let mut table_buf = String::new();
    for (in_table, text) in &dropped {
        if *in_table {
            table_buf.push_str(text);
            table_buf.push('\n');
        } else {
            out.push_str(text);
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
        assert_eq!(alnum_stream("Hello, World! £5"), "helloworld5");
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
