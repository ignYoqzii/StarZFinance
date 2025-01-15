using System.IO;

namespace StarZFinance.Classes
{
    // This class handles the Config.txt file
    public static class ConfigManager
    {
        // Default values for the config
        private static readonly Dictionary<string, object> DefaultSettings = new()
        {
            { "DiscordRPC", true },
            { "DiscordRPCIdleStatus", "In the launcher" },
            { "Theme", "LightTheme" }
        };

        private static readonly string configFilePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "StarZ Finance", "Config.txt");
        private static readonly Dictionary<string, object> settings = new(DefaultSettings);

        static ConfigManager()
        {
            if (!File.Exists(configFilePath))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(configFilePath)!);
                File.Create(configFilePath).Close();
                WriteSettingsToFile(DefaultSettings);
            }
            else
            {
                LoadSettingsFromFile();
            }
        }

        private static void WriteSettingsToFile(Dictionary<string, object> settingsToWrite)
        {
            using StreamWriter writer = new(configFilePath);
            foreach (var kvp in settingsToWrite)
            {
                writer.WriteLine($"{kvp.Key} = {kvp.Value}");
            }
        }

        private static void LoadSettingsFromFile()
        {
            if (!File.Exists(configFilePath)) return;

            using StreamReader reader = new(configFilePath);
            string line;
            while ((line = reader.ReadLine()!) != null)
            {
                var parts = line.Split('=').Select(part => part.Trim()).ToArray();
                if (parts.Length == 2 && settings.ContainsKey(parts[0]))
                {
                    var key = parts[0];
                    var value = parts[1];

                    settings[key] = ConvertValue(settings[key].GetType(), value);
                }
            }
        }

        private static object ConvertValue(Type targetType, string value)
        {
            if (targetType == typeof(bool))
            {
                return bool.Parse(value);
            }
            if (targetType == typeof(string))
            {
                return value;
            }
            throw new InvalidOperationException($"Unsupported type {targetType}");
        }

        private static void UpdateSetting(string key, object newValue)
        {
            if (settings.ContainsKey(key))
            {
                settings[key] = newValue;
                WriteSettingsToFile(settings);
            }
        }

        public static bool GetDiscordRPC() => (bool)settings["DiscordRPC"];
        public static string GetDiscordRPCIdleStatus() => (string)settings["DiscordRPCIdleStatus"];
        public static string GetTheme() => (string)settings["Theme"];
        
        public static void SetTheme(string newTheme) => UpdateSetting("Theme", newTheme);
        public static void SetDiscordRPC(bool value) => UpdateSetting("DiscordRPC", value);
        public static void SetDiscordRPCIdleStatus(string newIdleStatus) => UpdateSetting("DiscordRPCIdleStatus", newIdleStatus);
    }
}