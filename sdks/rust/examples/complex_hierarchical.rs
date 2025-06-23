// Complex hierarchical statechart example demonstrating deep nesting,
// multiple levels of states, and sophisticated state management patterns.

use statecharts::factory::*;
use statecharts::v1::*;

fn main() {
    println!("🏗️  Creating Complex Hierarchical Statechart Example");
    
    // Create a complex UI application state machine with deep hierarchy
    let app_statechart = create_application_statechart();
    
    // Validate and display the statechart
    validate_statechart(&app_statechart);
    display_statechart_structure(&app_statechart);
    
    println!("✅ Complex hierarchical statechart created successfully!");
}

fn create_application_statechart() -> Statechart {
    // Create deep nested authentication states
    let logged_out = basic_state("LoggedOut", true);
    let logging_in = basic_state("LoggingIn", false);
    let authenticating = basic_state("Authenticating", false);
    let logged_in = basic_state("LoggedIn", false);
    
    let auth_state = normal_state("Authentication", true, vec![
        logged_out, logging_in, authenticating, logged_in
    ]);
    
    // Create main application states with nested views
    let dashboard_overview = basic_state("Overview", true);
    let dashboard_analytics = basic_state("Analytics", false);
    let dashboard_reports = basic_state("Reports", false);
    
    let dashboard = normal_state("Dashboard", true, vec![
        dashboard_overview, dashboard_analytics, dashboard_reports
    ]);
    
    // Create user management nested states
    let user_list = basic_state("UserList", true);
    let user_details = basic_state("UserDetails", false);
    let user_edit = basic_state("UserEdit", false);
    let user_create = basic_state("UserCreate", false);
    
    let user_mgmt = normal_state("UserManagement", false, vec![
        user_list, user_details, user_edit, user_create
    ]);
    
    // Create settings with multiple sub-sections
    let profile_settings = basic_state("Profile", true);
    let security_settings = basic_state("Security", false);
    let notification_settings = basic_state("Notifications", false);
    
    let settings = normal_state("Settings", false, vec![
        profile_settings, security_settings, notification_settings
    ]);
    
    // Main application state containing all views
    let main_app = normal_state("MainApplication", false, vec![
        dashboard, user_mgmt, settings
    ]);
    
    // Create loading states
    let initial_loading = basic_state("InitialLoading", true);
    let data_loading = basic_state("DataLoading", false);
    
    let loading = normal_state("Loading", false, vec![
        initial_loading, data_loading
    ]);
    
    // Create error states
    let network_error = basic_state("NetworkError", true);
    let auth_error = basic_state("AuthError", false);
    let general_error = basic_state("GeneralError", false);
    
    let error_state = normal_state("Error", false, vec![
        network_error, auth_error, general_error
    ]);
    
    // Root application state
    let root_state = normal_state("Application", true, vec![
        auth_state, main_app, loading, error_state
    ]);
    
    // Create comprehensive transitions
    let mut statechart = statechart(root_state);
    
    // Authentication flow transitions
    statechart.transitions.extend(vec![
        transition("StartLogin", vec!["LoggedOut"], vec!["LoggingIn"], "LOGIN_ATTEMPT"),
        transition("Authenticate", vec!["LoggingIn"], vec!["Authenticating"], "SUBMIT_CREDENTIALS"),
        transition("LoginSuccess", vec!["Authenticating"], vec!["LoggedIn", "MainApplication", "Dashboard", "Overview"], "AUTH_SUCCESS"),
        transition("LoginFailure", vec!["Authenticating"], vec!["LoggedOut"], "AUTH_FAILURE"),
        transition("Logout", vec!["LoggedIn"], vec!["LoggedOut"], "LOGOUT"),
    ]);
    
    // Main application navigation transitions
    statechart.transitions.extend(vec![
        transition("GoToDashboard", vec!["UserManagement", "Settings"], vec!["Dashboard", "Overview"], "NAV_DASHBOARD"),
        transition("GoToUsers", vec!["Dashboard", "Settings"], vec!["UserManagement", "UserList"], "NAV_USERS"),
        transition("GoToSettings", vec!["Dashboard", "UserManagement"], vec!["Settings", "Profile"], "NAV_SETTINGS"),
    ]);
    
    // Dashboard sub-navigation
    statechart.transitions.extend(vec![
        transition("ViewAnalytics", vec!["Overview", "Reports"], vec!["Analytics"], "VIEW_ANALYTICS"),
        transition("ViewReports", vec!["Overview", "Analytics"], vec!["Reports"], "VIEW_REPORTS"),
        transition("ViewOverview", vec!["Analytics", "Reports"], vec!["Overview"], "VIEW_OVERVIEW"),
    ]);
    
    // User management workflow
    statechart.transitions.extend(vec![
        transition("SelectUser", vec!["UserList"], vec!["UserDetails"], "USER_SELECTED"),
        transition("EditUser", vec!["UserDetails"], vec!["UserEdit"], "EDIT_USER"),
        transition("CreateUser", vec!["UserList"], vec!["UserCreate"], "CREATE_USER"),
        transition("SaveUser", vec!["UserEdit", "UserCreate"], vec!["UserDetails"], "SAVE_USER"),
        transition("CancelEdit", vec!["UserEdit", "UserCreate"], vec!["UserList"], "CANCEL_EDIT"),
        transition("BackToList", vec!["UserDetails"], vec!["UserList"], "BACK_TO_LIST"),
    ]);
    
    // Settings navigation
    statechart.transitions.extend(vec![
        transition("ViewSecurity", vec!["Profile", "Notifications"], vec!["Security"], "VIEW_SECURITY"),
        transition("ViewNotifications", vec!["Profile", "Security"], vec!["Notifications"], "VIEW_NOTIFICATIONS"),
        transition("ViewProfile", vec!["Security", "Notifications"], vec!["Profile"], "VIEW_PROFILE"),
    ]);
    
    // Loading and error handling
    statechart.transitions.extend(vec![
        transition("StartLoading", vec!["MainApplication"], vec!["Loading", "DataLoading"], "START_LOADING"),
        transition("LoadingComplete", vec!["Loading"], vec!["MainApplication"], "LOADING_COMPLETE"),
        transition("NetworkError", vec!["MainApplication", "Loading"], vec!["Error", "NetworkError"], "NETWORK_ERROR"),
        transition("AuthError", vec!["MainApplication", "Loading"], vec!["Error", "AuthError"], "AUTH_ERROR"),
        transition("GeneralError", vec!["MainApplication", "Loading"], vec!["Error", "GeneralError"], "GENERAL_ERROR"),
        transition("RetryFromError", vec!["Error"], vec!["Loading", "InitialLoading"], "RETRY"),
        transition("RecoverFromError", vec!["Error"], vec!["MainApplication"], "RECOVER"),
    ]);
    
    // Define all events
    statechart.events.extend(vec![
        event("LOGIN_ATTEMPT"),
        event("SUBMIT_CREDENTIALS"),
        event("AUTH_SUCCESS"),
        event("AUTH_FAILURE"),
        event("LOGOUT"),
        event("NAV_DASHBOARD"),
        event("NAV_USERS"),
        event("NAV_SETTINGS"),
        event("VIEW_ANALYTICS"),
        event("VIEW_REPORTS"),
        event("VIEW_OVERVIEW"),
        event("USER_SELECTED"),
        event("EDIT_USER"),
        event("CREATE_USER"),
        event("SAVE_USER"),
        event("CANCEL_EDIT"),
        event("BACK_TO_LIST"),
        event("VIEW_SECURITY"),
        event("VIEW_NOTIFICATIONS"),
        event("VIEW_PROFILE"),
        event("START_LOADING"),
        event("LOADING_COMPLETE"),
        event("NETWORK_ERROR"),
        event("AUTH_ERROR"),
        event("GENERAL_ERROR"),
        event("RETRY"),
        event("RECOVER"),
    ]);
    
    statechart
}

fn validate_statechart(statechart: &Statechart) {
    if let Some(root) = &statechart.root_state {
        println!("📊 Statechart Validation:");
        println!("   Root State: {}", root.label);
        println!("   Total States: {}", count_states(root));
        println!("   Total Transitions: {}", statechart.transitions.len());
        println!("   Total Events: {}", statechart.events.len());
        println!("   Max Depth: {}", calculate_max_depth(root));
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

fn display_statechart_structure(statechart: &Statechart) {
    println!("🌳 Statechart Structure:");
    if let Some(root) = &statechart.root_state {
        display_state_tree(root, 0);
    }
    
    println!("\n🔄 Transition Summary:");
    for (i, transition) in statechart.transitions.iter().enumerate() {
        println!("   {}. {} -> {} on {}", 
                 i + 1, 
                 transition.from.join(", "), 
                 transition.to.join(", "), 
                 transition.event);
    }
}

fn display_state_tree(state: &statecharts::v1::State, depth: usize) {
    let indent = "  ".repeat(depth);
    let state_type = match state.r#type {
        0 => "UNSPECIFIED",
        1 => "BASIC",
        2 => "NORMAL",
        3 => "PARALLEL",
        _ => "UNKNOWN",
    };
    
    let initial_marker = if state.is_initial { " (initial)" } else { "" };
    
    println!("{}├─ {} [{}]{}", indent, state.label, state_type, initial_marker);
    
    for child in &state.children {
        display_state_tree(child, depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_hierarchical_creation() {
        let statechart = create_application_statechart();
        
        // Verify root state exists
        assert!(statechart.root_state.is_some());
        let root = statechart.root_state.as_ref().unwrap();
        assert_eq!(root.label, "Application");
        
        // Verify we have the expected number of main states
        assert_eq!(root.children.len(), 4); // Auth, MainApp, Loading, Error
        
        // Verify we have transitions
        assert!(!statechart.transitions.is_empty());
        assert!(statechart.transitions.len() > 20);
        
        // Verify we have events
        assert!(!statechart.events.is_empty());
        assert!(statechart.events.len() > 15);
    }

    #[test]
    fn test_state_depth_calculation() {
        let statechart = create_application_statechart();
        let root = statechart.root_state.as_ref().unwrap();
        
        let depth = calculate_max_depth(root);
        assert!(depth >= 4); // Should have at least 4 levels deep
    }

    #[test]
    fn test_state_counting() {
        let statechart = create_application_statechart();
        let root = statechart.root_state.as_ref().unwrap();
        
        let state_count = count_states(root);
        assert!(state_count > 20); // Should have more than 20 states total
    }
}