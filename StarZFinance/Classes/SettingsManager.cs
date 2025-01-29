using StarZFinance.Windows;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;

namespace StarZFinance.Classes
{
    public class SettingsManager
    {
        public static ObservableCollection<Setting>? AppSettings { get; private set; }

        public SettingsManager()
        {
            AppSettings =
            [
                new Setting
                {
                    Name = "Discord Rich Presence",
                    Description = "Enable or disable Discord Rich Presence for the application.",
                    Type = SettingType.CheckBox,
                    DefaultValue = ConfigManager.GetDiscordRPC(),
                    Action = SetDiscordRPC
                },
                new Setting
                {
                    Name = "Discord Rich Presence Idle Status",
                    Description = "Change your Discord status when idling in the application.",
                    Type = SettingType.TextBox,
                    DefaultValue = ConfigManager.GetDiscordRPCIdleStatus(),
                    Action = SetDiscordRPCIdleStatus
                },
                new Setting
                {
                    Name = "Open Application Folder",
                    Description = "Opens the folder where app resources are stored.",
                    Type = SettingType.Button,
                    DefaultValue = "Open", // Button's text
                    Action = (value) => OpenAppFolder()
                },
                new Setting
                {
                    Name = "Use OpenAI", // For future implementation
                    Description = "Uses ChatGPT from OpenAI to analyse the news and optimizes the prediction's accuracy.",
                    Type = SettingType.CheckBox,
                    DefaultValue = true
                }
            ];
        }

        /// <summary>
        /// Logic for handling specific settings
        /// </summary>
        
        private void OpenAppFolder()
        {
            string folderPath = Directory.CreateDirectory(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "StarZ Finance")).FullName;
            Process.Start(new ProcessStartInfo
            {
                FileName = folderPath,
                UseShellExecute = true
            });
        }

        private void SetDiscordRPC(object value)
        {
            if ((bool)value == true)
            {
                ConfigManager.SetDiscordRPC(true); // set DiscordRPC to true
                if (!DiscordRichPresenceManager.DiscordClient.IsInitialized)
                {
                    DiscordRichPresenceManager.DiscordClient.Initialize();
                }
                DiscordRichPresenceManager.SetPresence();
            }
            else
            {
                ConfigManager.SetDiscordRPC(false); // set DiscordRPC to false
                DiscordRichPresenceManager.DiscordClient.ClearPresence();
            }
        }

        private void SetDiscordRPCIdleStatus(object status)
        {
            try
            {
                ConfigManager.SetDiscordRPCIdleStatus((string)status);

                bool isDiscordRPCEnabled = ConfigManager.GetDiscordRPC();
                if (isDiscordRPCEnabled)
                {
                    DiscordRichPresenceManager.IdlePresence((string)status);
                }
            }
            catch (Exception ex)
            {
                StarZMessageBox.ShowDialog($"Error updating Discord status: {ex.Message}", "Error !", false);
            }
        }
    }

    public class Setting
    {
        public string? Name { get; set; }
        public string? Description { get; set; }
        public SettingType Type { get; set; }
        public object? DefaultValue { get; set; }
        public Action<object>? Action { get; set; }
    }

    public enum SettingType
    {
        CheckBox,
        TextBox,
        Button
    }
}