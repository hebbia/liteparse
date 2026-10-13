//! Rejoin table cells whose text wraps onto several visual lines.
//!
//! Projection builds one line per y band, so a cell that wraps ("PRICE PER" /
//! "USER") is split across lines and interleaved with its neighbours. PDF
//! producers usually write a cell's lines back to back, so a text item that
//! directly follows a cell fragment in content-stream order, sits one line
//! below it and overlaps it horizontally is that cell's next line. Folding it
//! into the cell's first line keeps the cell's text together for the table and
//! paragraph passes. Producers that write row by row have no such successors,
//! so their lines pass through unchanged.

use super::tables::{TABLE_MIN_TRACK_GAP_FLOOR_PT, TABLE_MIN_TRACK_GAP_FONT_MULT, gutter_cells};
use crate::types::{ProjectedLine, TextItem};

/// Longest cell, in lines. Longer successor chains are a text column.
const MAX_CELL_LINES: usize = 6;
/// Max distance between a cell's consecutive lines, in line heights.
const MAX_LINE_STEP_HEIGHT_MULT: f32 = 1.6;
/// Lines below the current one searched for the next fragment.
const MAX_LINE_LOOKAHEAD: usize = 4;

/// Returns the page's lines with wrapped cells folded, or `None` when nothing
/// was folded.
pub(super) fn merge_stream_wrapped_cells(lines: &[ProjectedLine]) -> Option<Vec<ProjectedLine>> {
    let mut taken: Vec<Vec<bool>> = lines.iter().map(|l| vec![false; l.spans.len()]).collect();
    let mut folds: Vec<((usize, usize), Vec<(usize, usize)>)> = Vec::new();

    for i in 0..lines.len() {
        if !eligible(&lines[i]) {
            continue;
        }
        let mut row: Vec<(usize, Vec<(usize, usize)>, bool)> = Vec::new();
        for s in 0..lines[i].spans.len() {
            if taken[i][s] || lines[i].spans[s].text.trim().is_empty() {
                continue;
            }
            if let Some((chain, numeric_stack)) = follow_cell(lines, &taken, i, s) {
                row.push((s, chain, numeric_stack));
            }
        }
        // A numeric column continuing straight into the next row's value means
        // the producer wrote the table column by column; nothing on this line
        // is a wrapped cell.
        if row.is_empty() || row.iter().any(|(_, _, numeric_stack)| *numeric_stack) {
            continue;
        }
        let last = row
            .iter()
            .filter_map(|(_, c, _)| c.last().map(|&(j, _)| j))
            .max()
            .unwrap_or(i);
        if !(i..=last).any(|j| is_cell_row(&lines[j])) {
            continue;
        }
        if starts_new_row(lines, i, &row) {
            continue;
        }
        for (s, chain, _) in row {
            for &(j, t) in &chain {
                taken[j][t] = true;
            }
            taken[i][s] = true;
            folds.push(((i, s), chain));
        }
    }
    if folds.is_empty() {
        return None;
    }

    let mut out: Vec<ProjectedLine> = lines.to_vec();
    let mut removed: Vec<Vec<bool>> = lines.iter().map(|l| vec![false; l.spans.len()]).collect();
    let mut added: Vec<Vec<TextItem>> = vec![Vec::new(); lines.len()];
    let mut changed = vec![false; lines.len()];
    for ((i, s), chain) in &folds {
        let mut cell = lines[*i].spans[*s].clone();
        for &(j, t) in chain {
            append_fragment(&mut cell, &lines[j].spans[t]);
            removed[j][t] = true;
            changed[j] = true;
        }
        let target = anchor_line(lines, *i, chain);
        changed[*i] = true;
        changed[target] = true;
        if target == *i {
            out[*i].spans[*s] = cell;
        } else {
            removed[*i][*s] = true;
            cell.y = lines[target].bbox.y;
            added[target].push(cell);
        }
    }
    let mut result = Vec::with_capacity(out.len());
    for (((mut line, gone), extra), changed) in out.into_iter().zip(removed).zip(added).zip(changed)
    {
        if !changed {
            result.push(line);
            continue;
        }
        let mut k = 0;
        line.spans.retain(|_| {
            k += 1;
            !gone[k - 1]
        });
        line.spans.extend(extra);
        if line.spans.iter().all(|sp| sp.text.trim().is_empty()) {
            continue;
        }
        rebuild_line(&mut line);
        result.push(line);
    }
    Some(result)
}

/// True when a continuation line also starts a new cell under one of line
/// `i`'s single-line cells: that column moved on to the next row, so the
/// "continuations" are the next row's cells, not wrapped text.
fn starts_new_row(
    lines: &[ProjectedLine],
    i: usize,
    row: &[(usize, Vec<(usize, usize)>, bool)],
) -> bool {
    let owners: Vec<usize> = row.iter().map(|(s, _, _)| *s).collect();
    let fragments: Vec<(usize, usize)> =
        row.iter().flat_map(|(_, c, _)| c.iter().copied()).collect();
    let single: Vec<&TextItem> = lines[i]
        .spans
        .iter()
        .enumerate()
        .filter(|(s, sp)| !owners.contains(s) && !sp.text.trim().is_empty())
        .map(|(_, sp)| sp)
        .collect();
    let mut continuation_lines: Vec<usize> = fragments.iter().map(|&(j, _)| j).collect();
    continuation_lines.dedup();
    continuation_lines.iter().any(|&j| {
        lines[j].spans.iter().enumerate().any(|(t, sp)| {
            !fragments.contains(&(j, t))
                && !sp.text.trim().is_empty()
                && single
                    .iter()
                    .any(|o| sp.x < o.x + o.width.max(0.0) && sp.x + sp.width.max(0.0) > o.x)
        })
    })
}

/// The line that carries the folded cell: of the lines the cell spans, the
/// one with the most cells (its row's own line when the cell is vertically
/// centred around it), else the cell's first line.
fn anchor_line(lines: &[ProjectedLine], i: usize, chain: &[(usize, usize)]) -> usize {
    let last = chain.iter().map(|&(j, _)| j).max().unwrap_or(i);
    let cells = |j: usize| {
        let line = &lines[j];
        if !eligible(line) || line.region_path != lines[i].region_path || !is_cell_row(line) {
            0
        } else {
            gutter_cells(line).len()
        }
    };
    let mut best = i;
    for j in i..=last {
        if cells(j) > cells(best) {
            best = j;
        }
    }
    best
}

fn eligible(line: &ProjectedLine) -> bool {
    !line.rtl && !line.all_mono
}

/// A line with at least two cells separated by table-width gutters.
fn is_cell_row(line: &ProjectedLine) -> bool {
    let cells = gutter_cells(line);
    if cells.len() < 2 {
        return false;
    }
    let font_size = if line.dominant_font_size > 0.0 {
        line.dominant_font_size
    } else {
        line.bbox.height.max(1.0)
    };
    let min_gutter = (font_size * TABLE_MIN_TRACK_GAP_FONT_MULT).max(TABLE_MIN_TRACK_GAP_FLOOR_PT);
    cells
        .windows(2)
        .all(|w| w[1].start_x - w[0].end_x >= min_gutter)
}

/// The fragments that continue span `s` of line `i`, as (line, span) indices,
/// and whether two or more of the chain's fragments are numbers (separate
/// records, not one cell). `None` when it has none or the chain is too long
/// to be a cell.
fn follow_cell(
    lines: &[ProjectedLine],
    taken: &[Vec<bool>],
    i: usize,
    s: usize,
) -> Option<(Vec<(usize, usize)>, bool)> {
    let first = &lines[i].spans[s];
    // A cell opens with words; a number above words is a value over the next
    // row's label.
    if !has_letters(first) {
        return None;
    }
    let (_, mut end) = first.stream_order?;
    let neighbours: Vec<(f32, f32)> = lines[i]
        .spans
        .iter()
        .enumerate()
        .filter(|&(t, sp)| t != s && !sp.text.trim().is_empty())
        .map(|(_, sp)| (sp.x, sp.x + sp.width.max(0.0)))
        .collect();
    let (mut lo, mut hi) = (first.x, first.x + first.width.max(0.0));
    let mut cur = i;
    let mut chain: Vec<(usize, usize)> = Vec::new();
    let mut numeric = usize::from(!has_letters(first));
    let font = (&first.font_name, first.font_size);
    let start = first.stream_order?.0;
    let mid_cell = (i.saturating_sub(MAX_LINE_LOOKAHEAD)..i).any(|j| {
        let line = &lines[j];
        let step = lines[i].bbox.y - line.bbox.y;
        let height = line.bbox.height.max(lines[i].bbox.height).max(1.0);
        step > 0.0
            && step <= height * MAX_LINE_STEP_HEIGHT_MULT
            && line.spans.iter().any(|sp| {
                sp.stream_order.is_some_and(|(_, b)| b + 1 == start)
                    && sp.x < first.x + first.width.max(0.0)
                    && sp.x + sp.width.max(0.0) > first.x
            })
    });
    if mid_cell {
        return None;
    }
    loop {
        let next = (cur..lines.len().min(cur + 1 + MAX_LINE_LOOKAHEAD)).find_map(|j| {
            let line = &lines[j];
            if j != cur {
                let step = line.bbox.y - lines[cur].bbox.y;
                let height = line.bbox.height.max(lines[cur].bbox.height).max(1.0);
                if step <= 0.0 || step > height * MAX_LINE_STEP_HEIGHT_MULT {
                    return None;
                }
            } else if j == i {
                return None;
            }
            if !eligible(line) || line.region_path != lines[i].region_path {
                return None;
            }
            line.spans.iter().enumerate().find_map(|(t, sp)| {
                let same_font = base_font(&sp.font_name) == base_font(font.0)
                    && (sp.font_size.unwrap_or(0.0) - font.1.unwrap_or(0.0)).abs() <= 0.5;
                let starts_after = same_font && sp.stream_order.is_some_and(|(a, _)| a == end + 1);
                let right = sp.x + sp.width.max(0.0);
                // A wrapped line stays inside its own column; one running under
                // the neighbouring cells is prose below the table.
                let overlaps = sp.x < hi
                    && right > lo
                    && !neighbours.iter().any(|&(a, b)| sp.x < b && right > a);
                (starts_after && overlaps && !taken[j][t] && !chain.contains(&(j, t)))
                    .then_some((j, t))
            })
        });
        let Some((j, t)) = next else { break };
        // Two cells of one line both under the opening fragment: it is a
        // title spanning columns, not a cell.
        let (first_lo, first_hi) = (first.x, first.x + first.width.max(0.0));
        let under = lines[j]
            .spans
            .iter()
            .filter(|sp| {
                !sp.text.trim().is_empty() && sp.x < first_hi && sp.x + sp.width.max(0.0) > first_lo
            })
            .count();
        if under >= 2 {
            return None;
        }
        let sp = &lines[j].spans[t];
        end = sp.stream_order.map_or(end, |(_, b)| b);
        lo = lo.min(sp.x);
        hi = hi.max(sp.x + sp.width.max(0.0));
        numeric += usize::from(!has_letters(sp));
        chain.push((j, t));
        cur = j;
        if chain.len() >= MAX_CELL_LINES {
            return None;
        }
    }
    (!chain.is_empty()).then_some((chain, numeric >= 2))
}

/// Font name without a subset tag (`BCDJEE+SegoeUI` -> `SegoeUI`): one font
/// is often embedded as several subsets within a single table.
fn base_font(name: &Option<String>) -> Option<&str> {
    name.as_deref().map(|n| match n.split_once('+') {
        Some((tag, rest)) if tag.len() == 6 && tag.chars().all(|c| c.is_ascii_uppercase()) => rest,
        _ => n,
    })
}

fn has_letters(span: &TextItem) -> bool {
    span.text.chars().any(char::is_alphabetic)
}

fn append_fragment(cell: &mut TextItem, fragment: &TextItem) {
    let right = (cell.x + cell.width).max(fragment.x + fragment.width);
    cell.x = cell.x.min(fragment.x);
    cell.width = right - cell.x;
    let text = cell.text.trim_end().to_string();
    cell.text = format!("{text} {}", fragment.text.trim());
    cell.absorb_stream_order(fragment);
    cell.words.extend(fragment.words.iter().cloned());
}

/// Re-derive a line's text and horizontal extent from its spans.
fn rebuild_line(line: &mut ProjectedLine) {
    line.spans.sort_by(|a, b| a.x.total_cmp(&b.x));
    let mut text = String::new();
    for sp in &line.spans {
        let piece = sp.text.trim();
        if piece.is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(piece);
    }
    let left = line
        .spans
        .iter()
        .map(|sp| sp.x)
        .fold(f32::INFINITY, f32::min);
    let right = line
        .spans
        .iter()
        .map(|sp| sp.x + sp.width.max(0.0))
        .fold(f32::NEG_INFINITY, f32::max);
    if left.is_finite() && right > left {
        line.bbox.x = left;
        line.bbox.width = right - left;
        line.indent_x = left;
    }
    line.text = text;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown_layout::test_helpers::line_with_spans;

    /// Visual lines 12pt apart whose spans carry the given stream indices.
    fn ordered_lines(rows: &[&[(&str, f32, u32)]]) -> Vec<ProjectedLine> {
        rows.iter()
            .enumerate()
            .map(|(r, row)| {
                let cells: Vec<(&str, f32)> = row.iter().map(|&(t, x, _)| (t, x)).collect();
                let mut line = line_with_spans(&cells, 100.0 + 12.0 * r as f32, 10.0);
                for (span, &(_, _, order)) in line.spans.iter_mut().zip(row.iter()) {
                    span.stream_order = Some((order, order));
                }
                line
            })
            .collect()
    }

    fn texts(line: &ProjectedLine) -> Vec<&str> {
        line.spans.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn cells_written_back_to_back_fold_into_their_first_line() {
        let lines = ordered_lines(&[
            &[
                ("PRODUCT", 40.0, 0),
                ("ANNUAL", 200.0, 1),
                ("NUMBER OF", 320.0, 4),
            ],
            &[("PRICE PER", 200.0, 2), ("USERS", 320.0, 5)],
            &[("USER", 200.0, 3)],
            &[
                ("Educate", 40.0, 6),
                ("$20.40", 200.0, 7),
                ("1900", 320.0, 8),
            ],
        ]);
        let out = merge_stream_wrapped_cells(&lines).expect("folds");
        assert_eq!(out.len(), 2);
        assert_eq!(
            texts(&out[0]),
            ["PRODUCT", "ANNUAL PRICE PER USER", "NUMBER OF USERS"]
        );
        assert_eq!(out[0].text, "PRODUCT ANNUAL PRICE PER USER NUMBER OF USERS");
        assert_eq!(texts(&out[1]), ["Educate", "$20.40", "1900"]);
    }

    #[test]
    fn a_label_wrapping_beside_single_line_cells_folds() {
        let lines = ordered_lines(&[
            &[("EURIBOR", 40.0, 0)],
            &[("Floor", 40.0, 1), ("0%", 200.0, 2), ("0.75%", 320.0, 3)],
        ]);
        let out = merge_stream_wrapped_cells(&lines).expect("folds");
        assert_eq!(out.len(), 1);
        assert_eq!(texts(&out[0]), ["EURIBOR Floor", "0%", "0.75%"]);
    }

    #[test]
    fn a_header_whose_cells_all_wrap_twice_folds() {
        let lines = ordered_lines(&[
            &[
                ("% of", 40.0, 0),
                ("Base Rent", 200.0, 2),
                ("Lease", 320.0, 4),
            ],
            &[("NRA", 40.0, 1), ("PSF", 200.0, 3), ("Expiry", 320.0, 5)],
        ]);
        let out = merge_stream_wrapped_cells(&lines).expect("folds");
        assert_eq!(out.len(), 1);
        assert_eq!(
            texts(&out[0]),
            ["% of NRA", "Base Rent PSF", "Lease Expiry"]
        );
    }

    #[test]
    fn a_cell_centred_around_its_row_line_folds() {
        let mut lines = ordered_lines(&[
            &[("Vacated", 320.0, 3)],
            &[
                ("VACANT (fmr. Deloitte)", 40.0, 0),
                ("128,000", 200.0, 1),
                ("-", 420.0, 5),
            ],
            &[("12/2021", 320.0, 4)],
        ]);
        for (k, line) in lines.iter_mut().enumerate() {
            line.bbox.y = 100.0 + 5.0 * k as f32;
        }
        let out = merge_stream_wrapped_cells(&lines).expect("folds");
        assert_eq!(out.len(), 1);
        assert_eq!(
            texts(&out[0]),
            ["VACANT (fmr. Deloitte)", "128,000", "Vacated 12/2021", "-"]
        );
    }

    #[test]
    fn a_section_label_in_another_font_does_not_absorb_the_next_row() {
        let mut lines = ordered_lines(&[
            &[("VALUATION RESULTS", 40.0, 0)],
            &[
                ("As-Is Value Conclusion", 40.0, 1),
                ("$40,400,000", 300.0, 2),
            ],
        ]);
        lines[0].spans[0].font_name = Some("Arial-Bold".into());
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn a_label_followed_by_the_next_rows_label_is_not_a_wrapped_cell() {
        let lines = ordered_lines(&[
            &[("Rate Timeframe", 40.0, 0), ("Monthly", 200.0, 2)],
            &[("Unit Type", 40.0, 1), ("No.", 200.0, 4)],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn a_note_below_the_table_is_not_its_last_cell() {
        let lines = ordered_lines(&[
            &[
                ("ACT Below 19", 40.0, 0),
                ("40%", 200.0, 1),
                ("42%", 320.0, 2),
            ],
            &[(
                "The cohort includes degree-seeking students who enroll",
                40.0,
                3,
            )],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn a_value_above_the_next_label_is_not_a_wrapped_cell() {
        let lines = ordered_lines(&[
            &[("23%", 40.0, 0), ("12%", 200.0, 3)],
            &[("Graduated from", 40.0, 1), ("Transferred to", 200.0, 4)],
            &[("Any College", 40.0, 2), ("Another Institution", 200.0, 5)],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn a_title_over_a_small_table_is_not_a_cell() {
        let lines = ordered_lines(&[
            &[("Coverage D Increased Limit (Loss of Use)", 60.0, 0)],
            &[("All Forms", 60.0, 1)],
            &[("Per", 120.0, 2), ("Rate", 200.0, 3)],
            &[("$1,000", 120.0, 4), ("$2.66", 200.0, 5)],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn rows_written_line_by_line_are_unchanged() {
        let lines = ordered_lines(&[
            &[
                ("PRODUCT", 40.0, 0),
                ("PRICE PER", 200.0, 1),
                ("NUMBER OF", 320.0, 2),
            ],
            &[
                ("PACKAGE", 40.0, 3),
                ("USER", 200.0, 4),
                ("USERS", 320.0, 5),
            ],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn a_single_text_column_is_not_a_cell() {
        let lines = ordered_lines(&[
            &[("the quick brown fox", 40.0, 0)],
            &[("jumps over the lazy", 40.0, 1)],
            &[("dog and keeps going", 40.0, 2)],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn long_column_chains_beside_a_cell_row_are_text_columns() {
        let lines = ordered_lines(&[
            &[("first column text", 40.0, 0), ("second column", 300.0, 10)],
            &[("more of the first", 40.0, 1), ("more second", 300.0, 11)],
            &[("still the first", 40.0, 2), ("still second", 300.0, 12)],
            &[("first goes on", 40.0, 3), ("second goes on", 300.0, 13)],
            &[("and on and on", 40.0, 4), ("and on again", 300.0, 14)],
            &[("further still", 40.0, 5), ("further again", 300.0, 15)],
            &[("the last line", 40.0, 6), ("the end", 300.0, 16)],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }

    #[test]
    fn numbers_written_column_by_column_stay_separate_rows() {
        let lines = ordered_lines(&[
            &[
                ("Net sales - US", 40.0, 0),
                ("147,007", 200.0, 3),
                ("140,083", 320.0, 6),
            ],
            &[
                ("Net sales - intl", 40.0, 1),
                ("12,507", 200.0, 4),
                ("12,586", 320.0, 7),
            ],
            &[
                ("Net sales", 40.0, 2),
                ("159,514", 200.0, 5),
                ("152,669", 320.0, 8),
            ],
        ]);
        assert!(merge_stream_wrapped_cells(&lines).is_none());
    }
}
