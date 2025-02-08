using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text.Json;

namespace StarZFinance.Classes
{
    public class LocalizationManager
    {
        private static LocalizationManager? _instance;
        private Dictionary<string, string>? localizedStrings;
        private string currentLanguage;

        public static LocalizationManager Instance => _instance ??= new LocalizationManager();

        private LocalizationManager()
        {
            currentLanguage = "en";  // Default language is English
            LoadLanguage(currentLanguage);
        }

        // Load translations based on the specified language (using embedded resources)
        public void LoadLanguage(string languageCode)
        {
            currentLanguage = languageCode;
            string resourceName = $"StarZFinance.Localization.{currentLanguage}.json";

            // Get the embedded resource stream
            var assembly = Assembly.GetExecutingAssembly();
            using Stream? stream = assembly.GetManifestResourceStream(resourceName)!;
            if (stream != null)
            {
                using StreamReader? reader = new(stream);
                string json = reader.ReadToEnd();
                localizedStrings = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
            }
            else
            {
                localizedStrings = [];
            }
        }

        // Get a translated string for a specific key
        public string GetString(string key)
        {
            return localizedStrings!.TryGetValue(key, out string? value) ? value : key;
        }

        // Get the current language code
        public string GetCurrentLanguage() => currentLanguage;
    }
}
