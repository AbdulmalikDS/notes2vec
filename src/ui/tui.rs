mod search;

use crate::core::config::Config;
use crate::core::error::{Error, Result};
use crate::indexing::discovery::discover_files;
use crate::search::model::{EmbeddingModel, EMBEDDING_MODEL_ID};
use crate::storage::state::{calculate_file_hash, get_file_modified_time, StateStore};
use crate::storage::vectors::{VectorEntry, VectorStore};
use search::{perform_search, parse_file_filter_query};
use crossterm::cursor;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use std::io;
use std::path::{Path, PathBuf};
use std::collections::HashSet;

// TUI configuration constants
const MAX_PREVIEW_LINES: usize = 200;           // Maximum lines to show in details preview
const MAX_RESULTS_DISPLAYED: usize = 5;         // Maximum number of results to display (re-exported from search module)

/// Screen states for the TUI flow
#[derive(PartialEq)]
enum Screen {
    Welcome,
    DirectorySelection,
    Search,
}

/// Interactive TUI search interface
pub struct SearchTui {
    // Screen state
    current_screen: Screen,
    
    // Search state
    query: String,
    results: Vec<(VectorEntry, f32)>,
    selected: usize,
    search_mode: bool, // true = typing query, false = browsing results
    
    // Directory selection state
    current_dir: PathBuf,
    dir_entries: Vec<PathBuf>,
    dir_selected: usize,
    previous_dir: Option<PathBuf>, // Track previous directory for Esc navigation
    
    // Core components
    config: Config,
    vector_store: Option<VectorStore>,
    model: Option<EmbeddingModel>,

    // UI status (short-lived messages shown in directory selection footer)
    status_message: Option<String>,

    // Whether the real embedding model is loaded (vs hash fallback)
    model_ready: bool,

    // Limit searches to the files discovered in the currently selected folder
    active_files: HashSet<String>,
}

impl SearchTui {
    pub fn new(config: Config) -> Result<Self> {
        // Start with welcome screen
        let current_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."));
        
        let (dir_entries, _) = Self::list_directory(&current_dir)?;

        Ok(Self {
            current_screen: Screen::Welcome,
            query: String::new(),
            results: Vec::new(),
            selected: 0,
            search_mode: true,
            current_dir,
            dir_entries,
            dir_selected: 0,
            previous_dir: None,
            config,
            vector_store: None,
            model: None,
            status_message: None,
            model_ready: false,
            active_files: HashSet::new(),
        })
    }
    
    /// List directory entries (directories and supported note files)
    fn list_directory(path: &Path) -> Result<(Vec<PathBuf>, usize)> {
        let mut entries = Vec::new();
        let mut selected = 0;
        
        if let Some(parent) = path.parent() {
            entries.push(parent.to_path_buf());
            selected = 1;
        }
        
        if path.is_dir() {
            let mut dirs = Vec::new();
            let mut files = Vec::new();
            
            match std::fs::read_dir(path) {
                Ok(entries_iter) => {
                    for entry in entries_iter {
                        if let Ok(entry) = entry {
                            let entry_path = entry.path();
                            if entry_path.is_dir() {
                                dirs.push(entry_path);
                            } else if entry_path.extension()
                                .and_then(|e| e.to_str())
                                .map(|e| matches!(e.to_lowercase().as_str(), "md" | "markdown" | "mdown" | "mkd" | "mkdn" | "txt"))
                                .unwrap_or(false) {
                                files.push(entry_path);
                            }
                        }
                    }
                }
                Err(_) => {}
            }
            
            // Sort: directories first, then files
            dirs.sort();
            files.sort();
            entries.extend(dirs);
            entries.extend(files);
        }
        
        Ok((entries, selected))
    }

    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture, cursor::Hide)?;

        // Always restore terminal, even if we early-return with an error.
        struct TerminalRestore;
        impl Drop for TerminalRestore {
            fn drop(&mut self) {
                let _ = disable_raw_mode();
                let mut stdout = io::stdout();
                let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture, cursor::Show);
            }
        }
        let _restore = TerminalRestore;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = ratatui::Terminal::new(backend)?;

        let mut should_quit = false;

        while !should_quit {
            terminal.draw(|f| self.render_ui(f))?;

            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match self.current_screen {
                        Screen::Welcome => {
                            match key.code {
                                KeyCode::Char('q') | KeyCode::Esc => {
                                    should_quit = true;
                                }
                                KeyCode::Enter => {
                                    // Reset previous_dir when entering directory selection from welcome
                                    self.previous_dir = None;
                                    self.current_screen = Screen::DirectorySelection;
                                }
                                _ => {}
                            }
                        }
                        Screen::DirectorySelection => {
                            match key.code {
                                KeyCode::Char('q') => {
                                    should_quit = true;
                                }
                                KeyCode::Esc => {
                                    // Go back one page: navigate up directory if possible, otherwise go to welcome
                                    if let Some(prev_dir) = &self.previous_dir {
                                        // Go back to previous directory
                                        self.current_dir = prev_dir.clone();
                                        self.previous_dir = self.current_dir.parent().map(|p| p.to_path_buf());
                                        if let Ok((entries, sel)) = Self::list_directory(&self.current_dir) {
                                            self.dir_entries = entries;
                                            self.dir_selected = sel;
                                        }
                                        self.status_message = None;
                                    } else {
                                        // No previous directory - go back to welcome screen
                                        self.current_screen = Screen::Welcome;
                                        self.status_message = None;
                                    }
                                }
                                KeyCode::Char('s') => {
                                    // Start searching in the current directory (only if it has note files)
                                    self.status_message = None;
                                    let current_dir_clone = self.current_dir.clone();
                                    self.select_directory(&current_dir_clone)?;
                                }
                                KeyCode::Enter => {
                                    self.status_message = None;
                                    // Enter: navigate/open (so user can browse folders and see .md/.txt)
                                    if self.dir_selected == 0 && self.current_dir.parent().is_some() {
                                        // ".." selected - navigate up
                                        if let Some(parent) = self.current_dir.parent() {
                                            // Save current directory as previous before navigating up
                                            self.previous_dir = Some(self.current_dir.clone());
                                            self.current_dir = parent.to_path_buf();
                                            if let Ok((entries, sel)) = Self::list_directory(&self.current_dir) {
                                                self.dir_entries = entries;
                                                self.dir_selected = sel;
                                            }
                                        }
                                    } else if let Some(selected_path) = self.dir_entries.get(self.dir_selected) {
                                        let selected_path_clone = selected_path.clone();
                                        if selected_path_clone.is_dir() {
                                            // Always navigate into directories on Enter
                                            // Save current directory as previous before navigating down
                                            self.previous_dir = Some(self.current_dir.clone());
                                            self.current_dir = selected_path_clone;
                                            if let Ok((entries, sel)) = Self::list_directory(&self.current_dir) {
                                                self.dir_entries = entries;
                                                self.dir_selected = sel;
                                            }
                                        } else {
                                            // Selected a note file: start search in the current directory
                                            let current_dir_clone = self.current_dir.clone();
                                            self.select_directory(&current_dir_clone)?;
                                        }
                                    } else {
                                        // No selection - treat as "search here"
                                        let current_dir_clone = self.current_dir.clone();
                                        self.select_directory(&current_dir_clone)?;
                                    }
                                }
                                KeyCode::Up => {
                                    self.status_message = None;
                                    if self.dir_selected > 0 {
                                        self.dir_selected -= 1;
                                    }
                                }
                                KeyCode::Down => {
                                    self.status_message = None;
                                    if self.dir_selected < self.dir_entries.len().saturating_sub(1) {
                                        self.dir_selected += 1;
                                    }
                                }
                                _ => {}
                            }
                        }
                        Screen::Search => {
                            match key.code {
                                KeyCode::Char('q') => {
                                    should_quit = true;
                                }
                                KeyCode::Esc => {
                                    // Always go back to directory selection
                                    self.current_screen = Screen::DirectorySelection;
                                    self.search_mode = true;
                                    self.query.clear();
                                    self.results.clear();
                                }
                                KeyCode::Enter if !self.search_mode => {
                                    // Enter edit mode (keep existing query so user can tweak it)
                                    self.search_mode = true;
                                }
                                KeyCode::Enter if self.search_mode => {
                                    self.perform_search()?;
                                    self.search_mode = false;
                                    self.selected = 0;
                                }
                                KeyCode::Char('u') if self.search_mode && key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    // Clear query
                                    self.query.clear();
                                }
                                KeyCode::Char(c) if self.search_mode => {
                                    self.query.push(c);
                                }
                                KeyCode::Char(c) if !self.search_mode => {
                                    // Start a new query quickly by just typing
                                    self.search_mode = true;
                                    self.query.clear();
                                    self.query.push(c);
                                }
                                KeyCode::Backspace if self.search_mode => {
                                    self.query.pop();
                                }
                                KeyCode::Up if !self.search_mode => {
                                    if self.selected > 0 {
                                        self.selected -= 1;
                                    }
                                }
                                KeyCode::Down if !self.search_mode => {
                                    if self.selected < self.results.len().saturating_sub(1) {
                                        self.selected += 1;
                                    }
                                }
                                KeyCode::Char('r') if !self.search_mode => {
                                    self.perform_search()?;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
    
    /// Select a directory and initialize search
    fn select_directory(&mut self, dir: &Path) -> Result<()> {
        // IMPORTANT: never let indexing/search setup errors kill the TUI loop.
        // We surface errors in the Directory Selection footer instead.
        let res: Result<()> = (|| {
            // TUI must not print while in raw/alternate screen mode.
            let model = EmbeddingModel::init_quiet(&self.config)?;
            self.model_ready = model.is_model_loaded();

            // Enforce real embeddings (no hash fallback) to keep results high-quality and avoid mixed indexes.
            if !self.model_ready {
                let model_path = self.config.models_dir.join("model.safetensors");
                let tokenizer_path = self.config.models_dir.join("tokenizer.json");
                let config_path = self.config.models_dir.join("config.json");
                self.status_message = Some(format!(
                    "Embedding model not loaded. Run: notes2vec init --base-dir {}  (missing: {}{}{})",
                    self.config.base_dir.display(),
                    if model_path.exists() { "" } else { "model.safetensors " },
                    if tokenizer_path.exists() { "" } else { "tokenizer.json " },
                    if config_path.exists() { "" } else { "config.json" },
                ));
                return Ok(());
            }

            // Open stores (after model is guaranteed)
            // If the model id changed, wipe stale indexes so results are consistent.
            let state_store = StateStore::open(&self.config)?;
            let previous_model_id = state_store.get_model_id()?.unwrap_or_default();
            drop(state_store);
            if previous_model_id != EMBEDDING_MODEL_ID {
                // Best-effort reset
                let _ = std::fs::remove_file(self.config.database_dir.join("vectors.redb"));
                let _ = std::fs::remove_file(&self.config.state_path);
            }

            let state_store = StateStore::open(&self.config)?;
            let vector_store = VectorStore::open(&self.config)?;

            // Discover and index files
            let files = discover_files(dir)?;
            if files.is_empty() {
                self.status_message = Some("No .md or .txt files found in this folder.".to_string());
                return Ok(());
            }

            // Scope searches to this folder's files (prevents showing results from other indexed folders)
            self.active_files.clear();
            for f in &files {
                if let Some(s) = f.relative_path.to_str() {
                    self.active_files.insert(s.to_string());
                }
            }

            for file in &files {
                // Convert path to string, skip if invalid UTF-8
                let file_path_str = match file.relative_path.to_str() {
                    Some(s) => s,
                    None => continue,
                };

                // Check if file has changed
                match (get_file_modified_time(&file.path), calculate_file_hash(&file.path)) {
                    (Ok(modified_time), Ok(hash)) => {
                        if state_store.has_file_changed(file_path_str, modified_time, &hash)? {
                            // Index the file
                            match crate::indexing::parser::parse_markdown_file(&file.path) {
                                Ok(doc) => {
                                    // Embed context + chunk text so headings like "Agenda" affect retrieval.
                                    let chunk_texts: Vec<String> = doc
                                        .chunks
                                        .iter()
                                        .map(|c| {
                                            if c.context.trim().is_empty() {
                                                c.text.clone()
                                            } else {
                                                format!("{}\n{}", c.context, c.text)
                                            }
                                        })
                                        .collect();

                                    let embeddings = model.embed_passages(&chunk_texts)?;
                                    for (chunk, embedding) in doc.chunks.iter().zip(embeddings.iter()) {
                                        let vector_entry = VectorEntry::new(
                                            file_path_str.to_string(),
                                            chunk.chunk_index,
                                            embedding.clone(),
                                            chunk.text.clone(),
                                            chunk.context.clone(),
                                            chunk.start_line,
                                            chunk.end_line,
                                        );
                                        let _ = vector_store.insert(&vector_entry);
                                    }
                                    let _ = state_store.update_file_state(file_path_str, modified_time, hash);
                                }
                                Err(_) => {}
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Record model id used for this index
            let _ = state_store.set_model_id(EMBEDDING_MODEL_ID);

            // Initialize search components
            self.vector_store = Some(vector_store);
            self.model = Some(model);
            self.current_screen = Screen::Search;
            self.status_message = None;

            Ok(())
        })();

        if let Err(e) = res {
            self.status_message = Some(format!("{}", e));
            // Stay in directory selection instead of exiting the whole app.
            self.current_screen = Screen::DirectorySelection;
            return Ok(());
        }

        Ok(())
    }

    fn perform_search(&mut self) -> Result<()> {
        let model = self.model.as_ref().ok_or_else(|| Error::Config("Model not initialized".to_string()))?;
        let vector_store = self.vector_store.as_ref().ok_or_else(|| Error::Config("Vector store not initialized".to_string()))?;

        let results = perform_search(&self.query, model, vector_store, &self.active_files)?;
        self.results = results;
        self.selected = 0;

        Ok(())
    }

    fn render_ui(&self, f: &mut Frame) {
        // Paint a consistent background so the UI doesn't depend on the user's terminal theme.
        // If the terminal doesn't support truecolor, this will be approximated.
        let size = f.size();
        let background = Block::default().style(Style::default().bg(Color::Rgb(35, 35, 35)));
        f.render_widget(background, size);

        match self.current_screen {
            Screen::Welcome => self.render_welcome(f),
            Screen::DirectorySelection => self.render_directory_selection(f),
            Screen::Search => self.render_search(f),
        }
    }
    
    fn render_welcome(&self, f: &mut Frame) {
        let size = f.size();

        // Welcome screen palette (RGB for consistency across terminals).
        let header_color = Color::Rgb(214, 175, 0); // gold-ish
        let art_color = Color::Rgb(80, 200, 200); // teal/cyan
        let muted = Color::Rgb(140, 140, 140);
        let key_enter = Color::Rgb(70, 200, 90);
        let key_quit = Color::Rgb(235, 90, 90);
        let key_bg = Color::Rgb(55, 55, 55);
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),     // Main area (left banner + right character)
                Constraint::Length(2),  // Footer (border + text)
            ])
            .split(size);

        // Split main area into left (text) and right (character)
        let main_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(52),
                Constraint::Length(2), // spacer
                Constraint::Percentage(46),
            ])
            .split(chunks[0]);

        // --- Adaptive logo (big only when there's room) ---
        let left_w = main_cols[0].width as usize;
        let big_logo: &[&str] = &[
            "███╗░░██╗░█████╗░████████╗███████╗░██████╗██████╗░██╗░░░██╗███████╗░█████╗░",
            "████╗░██║██╔══██╗╚══██╔══╝██╔════╝██╔════╝╚════██╗██║░░░██║██╔════╝██╔══██╗",
            "██╔██╗██║██║░░██║░░░██║░░░█████╗░░╚█████╗░░░███╔═╝╚██╗░██╔╝█████╗░░██║░░╚═╝",
            "██║╚████║██║░░██║░░░██║░░░██╔══╝░░░╚═══██╗██╔══╝░░░╚████╔╝░██╔══╝░░██║░░██╗",
            "██║░╚███║╚█████╔╝░░░██║░░░███████╗██████╔╝███████╗░░╚██╔╝░░███████╗╚█████╔╝",
            "╚═╝░░╚══╝░╚════╝░░░░╚═╝░░░╚══════╝╚═════╝░╚══════╝░░░╚═╝░░░╚══════╝░╚════╝░",
        ];
        let small_logo: &[&str] = &[
            "notes2vec",
            "Semantic search for your notes",
        ];
        let logo_lines: Vec<&str> = if left_w >= 78 { big_logo.to_vec() } else { small_logo.to_vec() };

        // Clip logo lines so they don't spill when the terminal is narrow.
        let max_left = left_w.saturating_sub(1);
        let mut header_lines: Vec<Line> = logo_lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let clipped: String = line.chars().take(max_left).collect();
                let style = if logo_lines.len() == 2 && i == 1 {
                    Style::default().fg(muted).add_modifier(Modifier::ITALIC)
                } else {
                    Style::default().fg(header_color).add_modifier(Modifier::BOLD)
                };
                Line::from(vec![Span::styled(clipped, style)])
            })
            .collect();

        // Vertically center header inside the left column when there's extra space.
        let left_h = main_cols[0].height as usize;
        if left_h > header_lines.len() {
            let pad_top = (left_h - header_lines.len()) / 2;
            let mut padded: Vec<Line> = Vec::with_capacity(left_h);
            padded.extend(std::iter::repeat(Line::from("")).take(pad_top));
            padded.append(&mut header_lines);
            header_lines = padded;
        }

        let header_paragraph = Paragraph::new(header_lines)
            .block(Block::default().borders(Borders::NONE))
            .alignment(Alignment::Left);

        f.render_widget(header_paragraph, main_cols[0]);

        // Character ASCII art (right): choose smaller art when space is tight
        // Auto-fits (crops) to the available terminal height so the footer stays visible.
        let big_character: &[&str] = &[
            "  |____________________________________________________|",
            "  | __     __   ____   ___ ||  ____    ____     _  __  |",
            "  ||  |__ |--|_| || |_|   |||_|**|*|__|+|+||___| ||  | |",
            "  ||==|^^||--| |=||=| |=*=||| |~~|~|  |=|=|| | |~||==| |",
            "  ||  |##||  | | || | |JRO|||-|  | |==|+|+||-|-|~||__| |",
            "  ||__|__||__|_|_||_|_|___|||_|__|_|__|_|_||_|_|_||__|_|",
            "  ||_______________________||__________________________|",
            "  | _____________________  ||      __   __  _  __    _ |",
            "  ||=|=|=|=|=|=|=|=|=|=|=| __..\\/|  |_|  ||#||==|  / /|",
            "  || | | | | | | | | | | |/\\ \\  \\\\|++|=|  || ||==| / / |",
            "  ||_|_|_|_|_|_|_|_|_|_|_/_/\\_.___\\__|_|__||_||__|/_/__|",
            "  |____________________ /\\~()/()~//\\ __________________|",
            "  | __   __    _  _     \\_  (_ .  _/ _    ___     _____|",
            "  ||~~|_|..|__| || |_ _   \\ //\\\\ /  |=|__|~|~|___| | | |",
            "  ||--|+|^^|==|1||2| | |__/\\ __ /\\__| |==|x|x|+|+|=|=|=|",
            "  ||__|_|__|__|_||_|_| /  \\ \\  / /  \\_|__|_|_|_|_|_|_|_|",
            "  |_________________ _/    \\/\\/\\/    \\_ _______________|",
            "  | _____   _   __  |/      \\../      \\|  __   __   ___|",
            "  ||_____|_| |_|##|_||   |   \\/ __|   ||_|==|_|++|_|-|||",
            "  ||______||=|#|--| |\\   \\   o    /   /| |  |~|  | | |||",
            "  ||______||_|_|__|_|_\\   \\  o   /   /_|_|__|_|__|_|_|||",
            "  |_________ __________\\___\\____/___/___________ ______|",
            "  |__    _  /    ________     ______           /| _ _ _|",
            "  |\\ \\  |=|/   //    /| //   /  /  / |        / ||%|%|%|",
            "  | \\/\\ |*/  .//____//.//   /__/__/ (_)      /  ||=|=|=|",
            "__|  \\/\\|/   /(____|/ //                    /  /||~|~|~|__",
            "  |___\\_/   /________//   ________         /  / ||_|_|_|",
            "  |___ /   (|________/   |\\_______\\       /  /| |______|",
            "      /                  \\|________)     /  / | |",
        ];
        let small_character: &[&str] = &[
            "                 .--.   _",
            "             .---|__| .((\\=.",
            "          .--|===|--|/    ,(,",
            "          |  |===|  |\\      y",
            "          |%%|   |  | `.__,'",
            "          |%%|   |  | /  \\\\\\",
            "          |  |   |  |/|  | \\`----.",
            "          |  |   |  ||\\  \\  |___.'_", 
            "         _|  |   |__||,\\  \\-+-._.' )_",
            "        / |  |===|--|\\  \\  \\      /  \\",
            "      /  `--^---'--' `--`-'---^-'    \\",
            "      '================================`",
        ];

        let right_w = main_cols[2].width as usize;
        let right_h = main_cols[2].height as usize;
        let character = if right_w < 62 || right_h < 18 { small_character } else { big_character };

        let available_lines = main_cols[2].height as usize;
        let mut start = 0usize;
        let mut end = character.len();
        if end > available_lines {
            let drop = end - available_lines;
            let drop_top = drop / 2;
            let drop_bottom = drop - drop_top;
            start = drop_top;
            end = end.saturating_sub(drop_bottom);
        }
        let slice = &character[start..end];

        let mut art_lines: Vec<Line> = Vec::new();
        // Vertically center character in the right column when there's extra space.
        if right_h > slice.len() {
            let pad_top = (right_h - slice.len()) / 2;
            art_lines.extend(std::iter::repeat(Line::from("")).take(pad_top));
        }

        art_lines.extend(
            slice
            .iter()
            .map(|line| {
                let max_right = right_w.saturating_sub(1);
                let clipped: String = line.chars().take(max_right).collect();
                Line::from(vec![Span::styled(
                    clipped,
                    Style::default().fg(art_color),
                )])
            }),
        );

        let art_paragraph = Paragraph::new(art_lines)
            .block(Block::default().borders(Borders::NONE))
            .alignment(Alignment::Center);

        f.render_widget(art_paragraph, main_cols[2]);

        // Footer "buttons" (key hints)
        let footer = Paragraph::new(Line::from(vec![
            Span::styled("[", Style::default().fg(muted)),
            Span::styled(
                " Enter ",
                Style::default().fg(key_enter).bg(key_bg).add_modifier(Modifier::BOLD),
            ),
            Span::styled("]", Style::default().fg(muted)),
            Span::raw(" Select Directory  "),
            Span::styled("•", Style::default().fg(muted)),
            Span::raw("  "),
            Span::styled("[", Style::default().fg(muted)),
            Span::styled(
                " q ",
                Style::default().fg(key_quit).bg(key_bg).add_modifier(Modifier::BOLD),
            ),
            Span::styled("]", Style::default().fg(muted)),
            Span::raw(" Quit"),
        ]))
        .style(Style::default().fg(muted))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(muted)),
        );

        f.render_widget(footer, chunks[1]);
    }
    
    fn render_directory_selection(&self, f: &mut Frame) {
        let size = f.size();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Title
                Constraint::Length(3), // Current path
                Constraint::Min(0),    // Directory list
                Constraint::Length(2), // Footer (increased for better visibility)
            ])
            .split(size);

        // Title (top-left)
        let title = Paragraph::new(Line::from(vec![Span::styled(
            "notes2vec",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]))
        .block(Block::default().borders(Borders::NONE))
        .alignment(Alignment::Left);
        f.render_widget(title, chunks[0]);

        // Current directory path
        let path_text = format!("Current directory: {}", self.current_dir.display());
        let path_para = Paragraph::new(path_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title("Select Notes Directory")
            )
            .style(Style::default().fg(Color::White));

        f.render_widget(path_para, chunks[1]);

        // Directory list
        let items: Vec<ListItem> = self.dir_entries
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let display_name = if i == 0 && self.current_dir.parent().is_some() {
                    ".. (parent directory)".to_string()
                } else if path.is_dir() {
                    format!("📁 {}", path.file_name().and_then(|n| n.to_str()).unwrap_or("?"))
                } else {
                    format!("📄 {}", path.file_name().and_then(|n| n.to_str()).unwrap_or("?"))
                };

                let style = if i == self.dir_selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                ListItem::new(Line::from(vec![Span::styled(display_name, style)]))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::White))
                    .title("Navigate")
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            );

        let mut list_state = ListState::default();
        list_state.select(Some(self.dir_selected));
        f.render_stateful_widget(list, chunks[2], &mut list_state);

        // Footer
        let mut footer_spans = vec![
            Span::styled("↑↓", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(": Navigate | "),
            Span::styled("Enter", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": Open | "),
            Span::styled("s", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": Search here | "),
            Span::styled("Esc", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(": Back | "),
            Span::styled("q", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(": Quit"),
        ];

        if let Some(msg) = &self.status_message {
            footer_spans.push(Span::raw("  |  "));
            footer_spans.push(Span::styled(msg.clone(), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)));
        }

        let footer = Paragraph::new(Line::from(footer_spans))
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(Color::DarkGray)));

        f.render_widget(footer, chunks[3]);
    }

    fn render_search(&self, f: &mut Frame) {
        let size = f.size();

        // Main layout: header with ASCII art, search bar, results, footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Title
                Constraint::Length(4), // Search bar
                Constraint::Min(0),    // Results
                Constraint::Length(2),  // Footer (border + text)
            ])
            .split(size);

        // Title (top-left)
        let title = Paragraph::new(Line::from(vec![Span::styled(
            "notes2vec",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]))
        .block(Block::default().borders(Borders::NONE))
        .alignment(Alignment::Left);
        f.render_widget(title, chunks[0]);

        // Search bar
        let search_block = Block::default()
            .borders(Borders::ALL)
            .border_style(if self.search_mode {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::White)
            })
            .title(vec![
                Span::styled("Search", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]);

        let search_text = if self.query.is_empty() {
            vec![Span::styled(
                "Type your search query...",
                Style::default().fg(Color::DarkGray),
            )]
        } else {
            vec![Span::styled(
                &self.query,
                Style::default().fg(Color::White),
            )]
        };

        let search_paragraph = Paragraph::new(Line::from(search_text))
            .block(search_block)
            .style(if self.search_mode {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });

        f.render_widget(search_paragraph, chunks[1]);

        // Results area
        if self.results.is_empty() {
            let empty_text = if self.query.is_empty() {
                vec![
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "Enter a search query above to find your notes",
                            Style::default().fg(Color::White),
                        ),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "Search by meaning, not just keywords",
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]),
                ]
            } else {
                vec![
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(
                            "No results found. Try a different query.",
                            Style::default().fg(Color::White),
                        ),
                    ]),
                    Line::from(""),
                ]
            };

            let empty_paragraph = Paragraph::new(empty_text)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::White))
                        .title("Results"),
                )
                .alignment(Alignment::Center);

            f.render_widget(empty_paragraph, chunks[2]);
        } else {
            // Split results area into list and details
            let result_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(40), // Results list
                    Constraint::Percentage(60), // Details panel
                ])
                .split(chunks[2]);

            // Results list
            let items: Vec<ListItem> = self
                .results
                .iter()
                .enumerate()
                .map(|(i, (entry, similarity))| {
                    let file_name = &entry.file_path;

                    let style = if i == self.selected {
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Yellow)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };

                    let similarity_pct = (similarity * 100.0) as u8;
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("[{:3}%] ", similarity_pct), style),
                        Span::styled(file_name.to_string(), style),
                    ]))
                })
                .collect();

            let list = List::new(items)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::White))
                        .title(vec![
                            Span::styled("Results", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                            Span::raw(format!(" ({} found)", self.results.len())),
                        ]),
                )
                .highlight_style(
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
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

        // Persistent footer "buttons" (always visible)
        let (file_filter, _semantic_query) = parse_file_filter_query(&self.query);
        let filter_note = file_filter
            .as_ref()
            .map(|f| format!("  Filter: {f}"))
            .unwrap_or_default();
        let model_note = format!("  Model: {}", EMBEDDING_MODEL_ID);
        let scope_note = format!("  Scope: {} ({} files)", self.current_dir.display(), self.active_files.len());
        let top_note = format!("  Top {} files", MAX_RESULTS_DISPLAYED);

        let footer_lines = if self.search_mode {
            vec![
                Line::from(vec![
                    Span::styled("Enter", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw(": Search  "),
                    Span::styled("Esc", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw(": Back  "),
                    Span::styled("Ctrl+U", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(": Clear  "),
                    Span::styled("q", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                    Span::raw(": Quit"),
                ]),
                Line::from(vec![
                    Span::styled("file:<name>", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::raw(": filter results"),
                    Span::raw(filter_note),
                    Span::raw(model_note),
                    Span::raw(top_note),
                    Span::raw(scope_note.clone()),
                ]),
            ]
        } else {
            vec![
                Line::from(vec![
                    Span::styled("↑↓", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(": Navigate  "),
                    Span::styled("Enter", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw(": Edit  "),
                    Span::styled("Esc", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw(": Back  "),
                    Span::styled("q", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                    Span::raw(": Quit"),
                ]),
                Line::from(vec![
                    Span::styled("file:<name>", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::raw(": filter results"),
                    Span::raw(filter_note),
                    Span::raw(model_note),
                    Span::raw(top_note),
                    Span::raw(scope_note.clone()),
                ]),
            ]
        };

        let footer = Paragraph::new(footer_lines)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );

        f.render_widget(footer, chunks[3]);
    }

    fn render_details<'a>(&self, entry: &'a VectorEntry, similarity: f32) -> Paragraph<'a> {
        let similarity_pct = (similarity * 100.0) as u8;
        let start_line = entry.start_line.max(1);
        let end_line = entry.end_line.max(start_line);

        let mut lines = vec![
            Line::from(vec![
                Span::styled("File: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(&entry.file_path, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Match: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(format!("{}%", similarity_pct), Style::default().fg(Color::Green)),
                Span::raw("  "),
                Span::styled("cos:", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:.3}", similarity), Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("Lines: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!("{}-{}", start_line, end_line),
                    Style::default().fg(Color::White),
                ),
            ]),
            if entry.context.trim().is_empty() {
                Line::from("")
            } else {
                Line::from(vec![
                    Span::styled("Context: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::styled(entry.context.clone(), Style::default().fg(Color::White)),
                ])
            },
            Line::from(""),
            Line::from(vec![
                Span::styled("Content:", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
        ];

        // Add content preview (truncate if too long)
        // Show a lot more content so the Details panel feels useful.
        let preview_lines: Vec<&str> = entry.text.lines().take(MAX_PREVIEW_LINES).collect();
        for line in preview_lines {
            lines.push(Line::from(vec![Span::styled(
                line.to_string(),
                Style::default().fg(Color::White),
            )]));
        }

        if entry.text.lines().count() > MAX_PREVIEW_LINES {
            lines.push(Line::from(vec![Span::styled(
                "... (truncated)",
                Style::default().fg(Color::DarkGray),
            )]));
        }

        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title("Details"),
            )
            .wrap(Wrap { trim: false })
    }
}