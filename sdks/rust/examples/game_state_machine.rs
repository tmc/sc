// Game state machine example demonstrating a complete game lifecycle
// with complex state transitions, parallel gameplay states, and comprehensive
// event handling for a modern video game.

use statecharts::factory::*;
use statecharts::v1::*;

fn main() {
    println!("🎮 Creating Game State Machine Example");
    
    let game_statechart = create_game_statechart();
    
    validate_and_display_game(&game_statechart);
    demonstrate_game_flow(&game_statechart);
    
    println!("✅ Game state machine created successfully!");
}

fn create_game_statechart() -> Statechart {
    // Main Menu States
    let main_menu = basic_state("MainMenu", true);
    let settings_menu = basic_state("SettingsMenu", false);
    let level_select = basic_state("LevelSelect", false);
    let character_select = basic_state("CharacterSelect", false);
    
    let menu_state = normal_state("MenuSystem", true, vec![
        main_menu, settings_menu, level_select, character_select
    ]);
    
    // Gameplay Core States
    let player_alive = basic_state("PlayerAlive", true);
    let player_dead = basic_state("PlayerDead", false);
    let player_invincible = basic_state("PlayerInvincible", false);
    
    let player_state = normal_state("PlayerState", true, vec![
        player_alive, player_dead, player_invincible
    ]);
    
    // Game World States
    let world_active = basic_state("WorldActive", true);
    let world_paused = basic_state("WorldPaused", false);
    let world_transitioning = basic_state("WorldTransitioning", false);
    
    let world_state = normal_state("WorldState", false, vec![
        world_active, world_paused, world_transitioning
    ]);
    
    // UI States (parallel with gameplay)
    let hud_visible = basic_state("HUDVisible", true);
    let hud_hidden = basic_state("HUDHidden", false);
    
    let ui_state = normal_state("UIState", false, vec![
        hud_visible, hud_hidden
    ]);
    
    // Audio States (parallel with gameplay)
    let music_playing = basic_state("MusicPlaying", true);
    let music_paused = basic_state("MusicPaused", false);
    let sfx_enabled = basic_state("SFXEnabled", true);
    let sfx_disabled = basic_state("SFXDisabled", false);
    
    let music_state = normal_state("MusicState", false, vec![
        music_playing, music_paused
    ]);
    
    let sfx_state = normal_state("SFXState", false, vec![
        sfx_enabled, sfx_disabled
    ]);
    
    let audio_state = parallel_state("AudioState", false, vec![
        music_state, sfx_state
    ]);
    
    // Main Gameplay (parallel regions for different game aspects)
    let gameplay = parallel_state("Gameplay", false, vec![
        player_state, world_state, ui_state, audio_state
    ]);
    
    // Loading States
    let loading_assets = basic_state("LoadingAssets", true);
    let loading_level = basic_state("LoadingLevel", false);
    let loading_save = basic_state("LoadingSave", false);
    
    let loading = normal_state("Loading", false, vec![
        loading_assets, loading_level, loading_save
    ]);
    
    // Game Over States
    let victory = basic_state("Victory", true);
    let defeat = basic_state("Defeat", false);
    let level_complete = basic_state("LevelComplete", false);
    
    let game_over = normal_state("GameOver", false, vec![
        victory, defeat, level_complete
    ]);
    
    // Pause States
    let pause_menu = basic_state("PauseMenu", true);
    let inventory = basic_state("Inventory", false);
    let options = basic_state("Options", false);
    
    let paused = normal_state("Paused", false, vec![
        pause_menu, inventory, options
    ]);
    
    // Root Game State
    let root_state = normal_state("Game", true, vec![
        menu_state, loading, gameplay, paused, game_over
    ]);
    
    let mut statechart = statechart(root_state);
    
    // Menu Navigation Transitions
    statechart.transitions.extend(vec![
        transition("OpenSettings", vec!["MainMenu"], vec!["SettingsMenu"], "OPEN_SETTINGS"),
        transition("OpenLevelSelect", vec!["MainMenu"], vec!["LevelSelect"], "NEW_GAME"),
        transition("BackToMain", vec!["SettingsMenu", "LevelSelect"], vec!["MainMenu"], "BACK_TO_MENU"),
        transition("SelectCharacter", vec!["LevelSelect"], vec!["CharacterSelect"], "SELECT_LEVEL"),
        transition("StartGame", vec!["CharacterSelect"], vec!["Loading", "LoadingLevel"], "START_GAME"),
    ]);
    
    // Game Lifecycle Transitions
    statechart.transitions.extend(vec![
        transition("GameLoaded", vec!["Loading"], vec!["Gameplay", "PlayerAlive", "WorldActive", "HUDVisible", "MusicPlaying", "SFXEnabled"], "GAME_LOADED"),
        transition("PauseGame", vec!["Gameplay"], vec!["Paused", "PauseMenu"], "PAUSE_GAME"),
        transition("ResumeGame", vec!["Paused"], vec!["Gameplay"], "RESUME_GAME"),
        transition("QuitToMenu", vec!["Paused", "GameOver"], vec!["MenuSystem", "MainMenu"], "QUIT_TO_MENU"),
    ]);
    
    // Player State Transitions
    statechart.transitions.extend(vec![
        transition("PlayerDies", vec!["PlayerAlive"], vec!["PlayerDead"], "PLAYER_DEATH"),
        transition("PlayerRespawn", vec!["PlayerDead"], vec!["PlayerInvincible"], "RESPAWN"),
        transition("InvincibilityEnd", vec!["PlayerInvincible"], vec!["PlayerAlive"], "INVINCIBLE_TIMEOUT"),
        transition("PowerUpCollected", vec!["PlayerAlive"], vec!["PlayerInvincible"], "POWER_UP"),
    ]);
    
    // World State Transitions
    statechart.transitions.extend(vec![
        transition("WorldPause", vec!["WorldActive"], vec!["WorldPaused"], "WORLD_PAUSE"),
        transition("WorldResume", vec!["WorldPaused"], vec!["WorldActive"], "WORLD_RESUME"),
        transition("StartTransition", vec!["WorldActive"], vec!["WorldTransitioning"], "LEVEL_TRANSITION"),
        transition("TransitionComplete", vec!["WorldTransitioning"], vec!["WorldActive"], "TRANSITION_COMPLETE"),
    ]);
    
    // UI State Transitions
    statechart.transitions.extend(vec![
        transition("HideHUD", vec!["HUDVisible"], vec!["HUDHidden"], "HIDE_HUD"),
        transition("ShowHUD", vec!["HUDHidden"], vec!["HUDVisible"], "SHOW_HUD"),
    ]);
    
    // Audio State Transitions
    statechart.transitions.extend(vec![
        transition("PauseMusic", vec!["MusicPlaying"], vec!["MusicPaused"], "PAUSE_MUSIC"),
        transition("ResumeMusic", vec!["MusicPaused"], vec!["MusicPlaying"], "RESUME_MUSIC"),
        transition("DisableSFX", vec!["SFXEnabled"], vec!["SFXDisabled"], "DISABLE_SFX"),
        transition("EnableSFX", vec!["SFXDisabled"], vec!["SFXEnabled"], "ENABLE_SFX"),
    ]);
    
    // Game Over Transitions
    statechart.transitions.extend(vec![
        transition("PlayerVictory", vec!["PlayerAlive"], vec!["GameOver", "Victory"], "VICTORY"),
        transition("PlayerDefeat", vec!["PlayerDead"], vec!["GameOver", "Defeat"], "DEFEAT"),
        transition("LevelCompleted", vec!["PlayerAlive"], vec!["GameOver", "LevelComplete"], "LEVEL_COMPLETE"),
        transition("RestartLevel", vec!["GameOver"], vec!["Loading", "LoadingLevel"], "RESTART"),
        transition("NextLevel", vec!["LevelComplete"], vec!["Loading", "LoadingLevel"], "NEXT_LEVEL"),
    ]);
    
    // Pause Menu Navigation
    statechart.transitions.extend(vec![
        transition("OpenInventory", vec!["PauseMenu"], vec!["Inventory"], "OPEN_INVENTORY"),
        transition("OpenOptions", vec!["PauseMenu"], vec!["Options"], "OPEN_OPTIONS"),
        transition("BackToPause", vec!["Inventory", "Options"], vec!["PauseMenu"], "BACK_TO_PAUSE"),
    ]);
    
    // Loading State Transitions
    statechart.transitions.extend(vec![
        transition("LoadAssets", vec!["LoadingLevel"], vec!["LoadingAssets"], "LOAD_ASSETS"),
        transition("LoadSave", vec!["LoadingAssets"], vec!["LoadingSave"], "LOAD_SAVE_DATA"),
        transition("LoadComplete", vec!["LoadingAssets", "LoadingSave"], vec!["Gameplay"], "LOADING_COMPLETE"),
    ]);
    
    // Define all game events
    statechart.events.extend(vec![
        // Menu events
        event("OPEN_SETTINGS"),
        event("NEW_GAME"),
        event("BACK_TO_MENU"),
        event("SELECT_LEVEL"),
        event("START_GAME"),
        
        // Game lifecycle events
        event("GAME_LOADED"),
        event("PAUSE_GAME"),
        event("RESUME_GAME"),
        event("QUIT_TO_MENU"),
        
        // Player events
        event("PLAYER_DEATH"),
        event("RESPAWN"),
        event("INVINCIBLE_TIMEOUT"),
        event("POWER_UP"),
        
        // World events
        event("WORLD_PAUSE"),
        event("WORLD_RESUME"),
        event("LEVEL_TRANSITION"),
        event("TRANSITION_COMPLETE"),
        
        // UI events
        event("HIDE_HUD"),
        event("SHOW_HUD"),
        
        // Audio events
        event("PAUSE_MUSIC"),
        event("RESUME_MUSIC"),
        event("DISABLE_SFX"),
        event("ENABLE_SFX"),
        
        // Game over events
        event("VICTORY"),
        event("DEFEAT"),
        event("LEVEL_COMPLETE"),
        event("RESTART"),
        event("NEXT_LEVEL"),
        
        // Pause menu events
        event("OPEN_INVENTORY"),
        event("OPEN_OPTIONS"),
        event("BACK_TO_PAUSE"),
        
        // Loading events
        event("LOAD_ASSETS"),
        event("LOAD_SAVE_DATA"),
        event("LOADING_COMPLETE"),
    ]);
    
    statechart
}

fn validate_and_display_game(statechart: &Statechart) {
    println!("🎲 Game State Machine Analysis:");
    
    if let Some(root) = &statechart.root_state {
        println!("   🎮 Game States: {}", count_states(root));
        println!("   🔄 Transitions: {}", statechart.transitions.len());
        println!("   📡 Events: {}", statechart.events.len());
        println!("   📊 Max Depth: {}", calculate_max_depth(root));
        println!("   🎭 Parallel Regions: {}", count_parallel_states(root));
    }
    
    println!("\n🏗️  Game Architecture:");
    display_game_structure(statechart);
}

fn demonstrate_game_flow(statechart: &Statechart) {
    println!("\n🎯 Typical Game Flow Demonstration:");
    
    let game_flow = vec![
        ("Game Start", "OPEN_SETTINGS"),
        ("Settings", "BACK_TO_MENU"),
        ("Main Menu", "NEW_GAME"),
        ("Level Select", "SELECT_LEVEL"),
        ("Character Select", "START_GAME"),
        ("Loading", "GAME_LOADED"),
        ("Gameplay", "PLAYER_DEATH"),
        ("Player Dead", "RESPAWN"),
        ("Player Invincible", "INVINCIBLE_TIMEOUT"),
        ("Player Alive", "PAUSE_GAME"),
        ("Paused", "OPEN_INVENTORY"),
        ("Inventory", "BACK_TO_PAUSE"),
        ("Pause Menu", "RESUME_GAME"),
        ("Gameplay", "VICTORY"),
        ("Game Over", "NEXT_LEVEL"),
        ("Loading", "QUIT_TO_MENU"),
    ];
    
    for (state, event) in game_flow {
        println!("   📍 {} --[{}]--> Next State", state, event);
    }
}

fn count_states(state: &statecharts::v1::State) -> usize {
    1 + state.children.iter().map(|child| count_states(child)).sum::<usize>()
}

fn calculate_max_depth(state: &statecharts::v1::State) -> usize {
    if state.children.is_empty() {
        1
    } else {
        1 + state.children.iter().map(|child| calculate_max_depth(child)).max().unwrap_or(0)
    }
}

fn count_parallel_states(state: &statecharts::v1::State) -> usize {
    let parallel_count = if state.r#type == 3 { 1 } else { 0 }; // 3 = PARALLEL
    parallel_count + state.children.iter().map(|child| count_parallel_states(child)).sum::<usize>()
}

fn display_game_structure(statechart: &Statechart) {
    if let Some(root) = &statechart.root_state {
        display_state_tree(root, 0);
    }
    
    println!("\n🔄 Critical Game Transitions:");
    let critical_events = vec![
        "START_GAME", "GAME_LOADED", "PAUSE_GAME", "RESUME_GAME", 
        "PLAYER_DEATH", "RESPAWN", "VICTORY", "DEFEAT"
    ];
    
    for transition in &statechart.transitions {
        if critical_events.contains(&transition.event.as_str()) {
            println!("   🎯 {} -> {} on {}", 
                     transition.from.join(", "), 
                     transition.to.join(", "), 
                     transition.event);
        }
    }
}

fn display_state_tree(state: &statecharts::v1::State, depth: usize) {
    let indent = "  ".repeat(depth);
    let state_type = match state.r#type {
        1 => "🟢", // BASIC
        2 => "🔵", // NORMAL
        3 => "🟡", // PARALLEL
        _ => "⚪",  // UNSPECIFIED
    };
    
    let initial_marker = if state.is_initial { " ⭐" } else { "" };
    
    println!("{}{} {}{}", indent, state_type, state.label, initial_marker);
    
    for child in &state.children {
        display_state_tree(child, depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_statechart_creation() {
        let statechart = create_game_statechart();
        
        assert!(statechart.root_state.is_some());
        let root = statechart.root_state.as_ref().unwrap();
        assert_eq!(root.label, "Game");
        
        // Should have main game sections
        assert_eq!(root.children.len(), 5); // Menu, Loading, Gameplay, Paused, GameOver
    }

    #[test]
    fn test_parallel_regions_exist() {
        let statechart = create_game_statechart();
        let root = statechart.root_state.as_ref().unwrap();
        
        let parallel_count = count_parallel_states(root);
        assert!(parallel_count > 0, "Game should have parallel states for audio/gameplay");
    }

    #[test]
    fn test_comprehensive_events() {
        let statechart = create_game_statechart();
        
        // Should have events for all major game systems
        let event_labels: Vec<String> = statechart.events.iter().map(|e| e.label.clone()).collect();
        
        assert!(event_labels.contains(&"START_GAME".to_string()));
        assert!(event_labels.contains(&"PLAYER_DEATH".to_string()));
        assert!(event_labels.contains(&"PAUSE_GAME".to_string()));
        assert!(event_labels.contains(&"VICTORY".to_string()));
        assert!(event_labels.contains(&"DEFEAT".to_string()));
    }

    #[test]
    fn test_transition_coverage() {
        let statechart = create_game_statechart();
        
        // Should have sufficient transitions for a complete game
        assert!(statechart.transitions.len() > 30, "Game should have comprehensive transitions");
    }

    #[test]
    fn test_state_depth() {
        let statechart = create_game_statechart();
        let root = statechart.root_state.as_ref().unwrap();
        
        let depth = calculate_max_depth(root);
        assert!(depth >= 3, "Game should have adequate state nesting");
    }
}