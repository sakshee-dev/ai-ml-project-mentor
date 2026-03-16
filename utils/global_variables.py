class GlobalVariables:
    """ This class is to store global variables that are shared between classes"""
    app_name = "AI ML Project Mentor"  # Initialization. Default name of the application
    version = "1.0.0"  # Initialization. Application version
    last_modified_date = "03/16/2026"  # Initialization. Application last modified date
    main_path = "C:\\Users\\Sakshee.Singh\\OneDrive - Nabors\\My Documents\\github_personal\\ai-ml-project-mentor"
    log_config_path = "C:\\Users\\Sakshee.Singh\\OneDrive - Nabors\\My Documents\\github_personal\\ai-ml-project-mentor\\logger_config.json"
    log_path = "C:\\Users\\Sakshee.Singh\\OneDrive - Nabors\\My Documents\\github_personal\\ai-ml-project-mentor\\log"
    # Exit codes
    LOCK_FILE_BUSY = 1
    MAPPING_DIR_NOT_EXIST = 2
    BAD_RECIPE_JSON = 3
    SETTING_NOT_EXIST = 4
    BAD_SETTING = 5
    BAD_PLC_TOPICS = 6
    DIVISION_BY_ZERO = 7
    SIGNAL_HEALTH_BAD = 8