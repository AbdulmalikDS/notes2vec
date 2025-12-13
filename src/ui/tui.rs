use crate::core::config::Config;
use crate::core::error::{Error, Result};
use crate::search::model::EmbeddingModel;
use crate::storage::vectors::{VectorEntry, VectorStore};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use std::io;

/// Interactive TUI search interface
pub struct SearchTui {
    query: String,
    results: Vec<(VectorEntry, f32)>,
    selected: usize,
    vector_store: VectorStore,
    model: EmbeddingModel,
}

impl SearchTui {
    pub fn new(config: Config) -> Result<Self> {
        let vector_store = VectorStore::open(&config)?;
        let model = EmbeddingModel::init(&config)?;

        Ok(Self {
            query: String::new(),
            results: Vec::new(),
            selected: 0,
            vector_store,
            model,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = ratatui::Terminal::new(backend)?;

        let mut should_quit = false;
        let mut search_mode = true; // true = typing query, false = browsing results

        while !should_quit {
            terminal.draw(|f| self.ui(f, search_mode))?;

            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            should_quit = true;
                        }
                        KeyCode::Enter if search_mode => {
                            self.perform_search()?;
                            search_mode = false;
                            self.selected = 0;
                        }
                        KeyCode::Char(c) if search_mode => {
                            self.query.push(c);
                        }
                        KeyCode::Backspace if search_mode => {
                            self.query.pop();
                        }
                        KeyCode::Up if !search_mode => {
                            if self.selected > 0 {
                                self.selected -= 1;
                            }
                        }
                        KeyCode::Down if !search_mode => {
                            if self.selected < self.results.len().saturating_sub(1) {
                                self.selected += 1;
                            }
                        }
                        KeyCode::Char('/') => {
                            search_mode = true;
                            self.query.clear();
                        }
                        KeyCode::Char('r') if !search_mode => {
                            self.perform_search()?;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn perform_search(&mut self) -> Result<()> {
        if self.query.trim().is_empty() {
            self.results.clear();
            return Ok(());
        }

        let query_texts = vec![self.query.clone()];
        let query_embeddings = self.model.embed(&query_texts)?;

        if query_embeddings.is_empty() {
            return Err(Error::Model("Failed to generate query embedding".to_string()));
        }

        let query_embedding = &query_embeddings[0];
        self.results = self.vector_store.search(query_embedding, 20)?;
        self.selected = 0;

        Ok(())
    }

    fn ui(&self, f: &mut Frame, search_mode: bool) {
        let size = f.size();

        // Main layout: search bar at top, results below
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Search bar
                Constraint::Min(0),    // Results
            ])
            .split(size);

        // Search bar
        let search_block = Block::default()
            .borders(Borders::ALL)
            .title(" Search ")
            .style(if search_mode {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });

        let search_text = if self.query.is_empty() {
            "Type your search query... (Press Enter to search, / to search again, q to quit)"
                .to_string()
        } else {
            self.query.clone()
        };

        let search_paragraph = Paragraph::new(search_text)
            .block(search_block)
            .wrap(Wrap { trim: true });

        f.render_widget(search_paragraph, chunks[0]);

        // Results area
        if self.results.is_empty() {
            let empty_text = if self.query.is_empty() {
                "Enter a search query above to find your notes"
            } else {
                "No results found. Try a different query."
            };

            let empty_block = Block::default()
                .borders(Borders::ALL)
                .title(" Results ");

            let empty_paragraph = Paragraph::new(empty_text)
                .block(empty_block)
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true });

            f.render_widget(empty_paragraph, chunks[1]);
        } else {
            // Split results area: list on left, details on right
            let result_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                .split(chunks[1]);

            // Results list
            let items: Vec<ListItem> = self
                .results
                .iter()
                .enumerate()
                .map(|(i, (entry, similarity))| {
                    let similarity_bar = format!("{:.1}%", similarity * 100.0);
                    let file_name = Path::new(&entry.file_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| entry.file_path.clone());

                    let style = if i == self.selected {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                    } else {
                        Style::default()
                    };

                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("{:.2} ", similarity),
                            Style::default().fg(Color::Cyan),
                        ),
                        Span::styled(file_name, style),
                        Span::raw(format!(" ({})", similarity_bar)),
                    ]))
                })
                .collect();

            let list = List::new(items)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" Results ({}) ", self.results.len())),
                )
                .highlight_style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                );

            let mut list_state = ListState::default();
            list_state.select(Some(self.selected));
            f.render_stateful_widget(list, result_chunks[0], &mut list_state);

            // Details panel
            if let Some((entry, similarity)) = self.results.get(self.selected) {
                let details = self.render_details(entry, *similarity);
                f.render_widget(details, result_chunks[1]);
            }
        }

        // Help text at bottom
        let help_text = if search_mode {
            "Enter: Search | Esc/q: Quit"
        } else {
            "↑↓: Navigate | /: New Search | r: Refresh | Esc/q: Quit"
        };

        let help = Paragraph::new(help_text)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);

        let help_area = Rect {
            x: 0,
            y: size.height.saturating_sub(1),
            width: size.width,
            height: 1,
        };

        f.render_widget(help, help_area);
    }

    fn render_details(&self, entry: &VectorEntry, similarity: f32) -> Paragraph<'static> {
        let file_name = Path::new(&entry.file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| entry.file_path.clone());

        let mut content = vec![
            Line::from(vec![
                Span::styled("File: ", Style::default().fg(Color::Cyan)),
                Span::styled(file_name.clone(), Style::default().add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("Similarity: ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("{:.1}%", similarity * 100.0),
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
        ];

        if !entry.context.is_empty() {
            content.push(Line::from(vec![
                Span::styled("Context: ", Style::default().fg(Color::Cyan)),
                Span::raw(entry.context.clone()),
            ]));
            content.push(Line::from(""));
        }

        content.push(Line::from(vec![
            Span::styled("Lines: ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}-{}", entry.start_line, entry.end_line)),
        ]));
        content.push(Line::from(""));
        content.push(Line::from(vec![Span::styled(
            "Content:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]));
        content.push(Line::from(""));

        // Wrap text content
        let text = &entry.text;
        let max_width = 60; // Approximate width for wrapping
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_line = String::new();

        for word in words {
            if current_line.len() + word.len() + 1 > max_width && !current_line.is_empty() {
                content.push(Line::from(current_line.clone()));
                current_line.clear();
            }
            if !current_line.is_empty() {
                current_line.push(' ');
            }
            current_line.push_str(word);
        }
        if !current_line.is_empty() {
            content.push(Line::from(current_line));
        }

        Paragraph::new(content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Details "),
            )
            .wrap(Wrap { trim: true })
    }
}

