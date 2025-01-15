using StarZFinance.Classes;
using System.Configuration;
using System.Data;
using System.IO;
using System.Net;
using System.Windows;

namespace StarZFinance
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
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
                string starZFinancePath = Path.Combine(documentsPath, "StarZ Finance");;
                string logsPath = Path.Combine(starZFinancePath, "Logs");

                EnsureDirectoryExists(starZFinancePath);
                EnsureDirectoryExists(logsPath);

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
                        LogsManager.Log("Could not initialize Discord Rich Presence. This is either because Discord RPC was disabled or Offline Mode is enabled.", logFileName);
                    }
                }
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error during application startup: {ex.Message}", logFileName);
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
                LogsManager.Log("Application exited successfully.", logFileName);
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error during application exit: {ex.Message}", logFileName);
            }
        }
    }

}
