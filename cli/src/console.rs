#[macro_export]
macro_rules! display_initialization {
    ($($arg:tt)*) => {
        println!(
            "{} {}",
            "///".dimmed(),
           format_args!($($arg)*)
        );
    };
}

#[macro_export]
macro_rules! display_info {
    ($($arg:tt)*) => {
         println!(
            "{} {}",
            "***".italic().bright_cyan(),
           format_args!($($arg)*)
        );
    };
}

#[macro_export]
macro_rules! display_success {
    ($($arg:tt)*) => {
        println!(
            "{} {}",
            ">>".bold().bright_green(),
           format_args!($($arg)*)
        );
    };
}

#[macro_export]
macro_rules! display_warning {
     ($($arg:tt)*) => {
         let message = format!($($arg)*);
         println!(
            "{} {}",
            "!>".bold().bright_yellow(),
            message.italic().bright_yellow()
        );
    };
}
