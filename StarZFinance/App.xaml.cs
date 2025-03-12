using StarZFinance.Classes;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;

namespace StarZFinance
{
    public partial class App : System.Windows.Application
    {

        private bool HasRun = false;
        private static readonly string logFileName = $"AppStartup.txt";

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            try
            {
                string documentsPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                string starzFinancePath = Path.Combine(documentsPath, "StarZ Finance"); ;
                string logsPath = Path.Combine(starzFinancePath, "Logs");
                string themePath = Path.Combine(starzFinancePath, "Theme");

                EnsureDirectoryExists(starzFinancePath);
                EnsureDirectoryExists(logsPath);
                EnsureDirectoryExists(themePath);
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error creating application's directory: {ex.Message}", logFileName);
            }

            try
            {
                if (!HasRun)
                {
                    bool discordRPC = ConfigManager.GetDiscordRPC();
                    if (discordRPC)
                    {
                        DiscordRichPresenceManager.DiscordClient.Initialize();
                        DiscordRichPresenceManager.SetPresence();
                        HasRun = true;
                        LogsManager.Log("Initialized Discord Rich Presence.", logFileName);
                    }
                    else
                    {
                        LogsManager.Log("Could not initialize Discord Rich Presence. Discord RPC is disabled", logFileName);
                    }
                }
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error initializing Discord Rich Presence: {ex.Message}", logFileName);
            }

            try
            {
                ThemesManager.SetAndInitializeThemes();
                LogsManager.Log("Themes loaded and initialized.", logFileName);
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error loading application's theme: {ex.Message}", logFileName);
            }
        }

        private static void EnsureDirectoryExists(string path)
        {
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
                LogsManager.Log($"Created directory: {path}", logFileName);
            }
        }

        private void App_OnExit(object sender, ExitEventArgs e)
        {
            try
            {
                DiscordRichPresenceManager.TerminatePresence();
                // More can be added
                LogsManager.Log("Application exited successfully.", logFileName);
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error during application's exit: {ex.Message}", logFileName);
            }
        }

        public static void CheckBoxAnimation(object sender)
        {
            if (sender is System.Windows.Controls.CheckBox checkBox)
            {
                // Access the ControlTemplate elements
                if (checkBox.Template.FindName("button", checkBox) is Border button)
                {
                    Storyboard Storyboard1 = (Storyboard)checkBox.FindResource("right");
                    Storyboard Storyboard2 = (Storyboard)checkBox.FindResource("left");
                    if (checkBox.IsChecked == true)
                    {
                        // Start the "right" animation when checked
                        Storyboard2.Stop(button);
                        Storyboard1.Begin(button);
                    }
                    else
                    {
                        // Start the "left" animation when unchecked
                        Storyboard1.Stop(button);
                        Storyboard2.Begin(button);
                    }
                }
            }
        }
    }
}
