using StarZFinance.Pages;
using StarZFinance.Windows;
using System.IO;
using System.Text.Json;
using System.Windows.Media;

namespace StarZFinance.Classes
{
    public static class ThemesManager
    {
        private static readonly string themeFilePath = Path.Combine(App.StarZFinanceDirectory, "Theme", "StarZTheme.szt");
        private static readonly string logFileName = "ThemesManager.txt";

        static ThemesManager()
        {
            if (!File.Exists(themeFilePath))
            {
                CreateDefaultThemes();
            }
        }

        private static void CreateDefaultThemes()
        {
            var defaultTheme = new
            {
                DarkTheme = CreateBaseTheme("#FF0044EA", "#FF00C7ED", "#FF1C1C1C", "#FF0D0D0D", "#FF0D0D0D", "#FFFFFFFF", "#FFFFFFFF", "#FFFFFFFF"),
                LightTheme = CreateBaseTheme("#FF0044EA", "#FF00C7ED", "#FFF3F4F9", "#FFFFFFFF", "#FF000000", "#FF000000", "#FFFFFFFF", "#FF000000"),
                CustomTheme = CreateBaseTheme("", "", "", "", "", "", "", "")
            };

            SaveThemesToFile(defaultTheme);
        }

        private static Dictionary<string, string> CreateBaseTheme(string accentColor1, string accentColor2, string primaryBackgroundColor,
            string secondaryBackgroundColor, string sideBarBackgroundColor, string iconColor, string sideBarIconColor, string textColor) =>
            new()
            {
                { "AccentColor1", accentColor1 },
                { "AccentColor2", accentColor2 },
                { "PrimaryBackgroundColor", primaryBackgroundColor },
                { "SecondaryBackgroundColor", secondaryBackgroundColor },
                { "SideBarBackgroundColor", sideBarBackgroundColor },
                { "IconColor", iconColor },
                { "SideBarIconColor", sideBarIconColor },
                { "TextColor", textColor }
            };

        private static void SaveThemesToFile(object themes)
        {
            try
            {
                var json = JsonSerializer.Serialize(themes, new JsonSerializerOptions { WriteIndented = true });
                Directory.CreateDirectory(Path.GetDirectoryName(themeFilePath)!);
                File.WriteAllText(themeFilePath, json);
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error saving themes: {ex.Message}", logFileName);
            }
        }

        private static SolidColorBrush GetSolidColorBrush(string colorKey) =>
            new SolidColorBrush((System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString(colorKey));

        public static void ApplyTheme(string themeName)
        {
            try
            {
                ConfigManager.SetTheme(themeName);
                var colors = LoadThemeColors(themeName);
                if (colors != null)
                {
                    UpdateApplicationResources(colors);
                    UpdateAccentGradientBrush(colors);
                    UpdateCheckBoxes(themeName);
                }
            }
            catch (Exception ex)
            {
                HandleThemeApplicationError(ex);
            }
        }

        private static void HandleThemeApplicationError(Exception ex)
        {
            LogsManager.Log($"Error applying theme: {ex.Message}", logFileName);
            bool? result = StarZMessageBox.ShowDialog("It seems that one or more themes contain invalid values, which has caused an error. Would you like to reset the themes? Click 'OK' to reset, or 'Cancel' to continue using the corrupted theme file.", "Warning !", true);
            if (result == true)
            {
                ResetThemesToDefault();
            }
        }

        private static void UpdateApplicationResources(Dictionary<string, string> colors)
        {
            foreach (var key in colors.Keys)
            {
                System.Windows.Application.Current.Resources[key] = GetSolidColorBrush(colors[key]);
            }
        }

        private static void UpdateAccentGradientBrush(Dictionary<string, string> colors)
        {
            var accentColor1 = (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString(colors["AccentColor1"]);
            var accentColor2 = (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString(colors["AccentColor2"]);

            var accentGradientBrush = new LinearGradientBrush
            {
                StartPoint = new System.Windows.Point(0.5, 0),
                EndPoint = new System.Windows.Point(0.5, 1),
                SpreadMethod = GradientSpreadMethod.Pad,
                MappingMode = BrushMappingMode.RelativeToBoundingBox
            };

            accentGradientBrush.GradientStops.Add(new GradientStop(accentColor1, 0.0));
            accentGradientBrush.GradientStops.Add(new GradientStop(MixColors(accentColor1, accentColor2), 0.5)); // Intermediate Color
            accentGradientBrush.GradientStops.Add(new GradientStop(accentColor2, 1.0));

            System.Windows.Application.Current.Resources["AccentColor"] = accentGradientBrush;
        }

        private static System.Windows.Media.Color MixColors(System.Windows.Media.Color color1, System.Windows.Media.Color color2) =>
            System.Windows.Media.Color.FromArgb(
                (byte)((color1.A + color2.A) / 2),
                (byte)((color1.R + color2.R) / 2),
                (byte)((color1.G + color2.G) / 2),
                (byte)((color1.B + color2.B) / 2));

        public static Dictionary<string, string>? LoadThemeColors(string themeName)
        {
            try
            {
                string json = File.ReadAllText(themeFilePath);
                var themes = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(json);
                return themes?.GetValueOrDefault(themeName);
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error loading theme colors: {ex.Message}", logFileName);
                return null;
            }
        }

        public static void UpdateThemeColorsInFile(string themeName, Dictionary<string, string> colors)
        {
            try
            {
                var themes = ReadThemesFromFile();
                if (themes != null)
                {
                    themes[themeName] = colors;
                    SaveThemesToFile(themes);
                }
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error updating theme colors: {ex.Message}", logFileName);
            }
        }

        private static Dictionary<string, Dictionary<string, string>>? ReadThemesFromFile()
        {
            try
            {
                string json = File.ReadAllText(themeFilePath);
                return JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(json);
            }
            catch
            {
                return null;
            }
        }

        public static void SetAndInitializeThemes() => ApplyTheme(ConfigManager.GetTheme()); // On Window loading

        public static void UpdateCheckBoxes(string theme)
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() =>
            {
                Settings.Instance?.UpdateCheckBoxes(theme);
            });
        }

        public static bool AreColorValuesValid(Dictionary<string, string> colors) =>
            colors.Values.All(value =>
            {
                if (string.IsNullOrEmpty(value)) return false;
                try
                {
                    System.Windows.Media.ColorConverter.ConvertFromString(value);
                    return true;
                }
                catch
                {
                    return false; // Invalid color format
                }
            });

        public static void ResetThemesToDefault()
        {
            try
            {
                CreateDefaultThemes();
                ApplyTheme("LightTheme");
            }
            catch (Exception ex)
            {
                LogsManager.Log($"Error resetting themes: {ex.Message}", logFileName);
            }
        }

        public static void ExportTheme(bool showSuccessMessage)
        {
            try
            {
                if (!File.Exists(themeFilePath))
                {
                    LogsManager.Log("No theme file found to backup.", logFileName);
                    return;
                }

                string? newName = EditWindow.ShowDialog("StarZTheme");
                if (newName != null)
                {
                    string backupFilePath = Path.Combine(Path.GetDirectoryName(themeFilePath) ?? string.Empty, newName + ".szt");
                    File.Copy(themeFilePath, backupFilePath, true);

                    if (showSuccessMessage)
                    {
                        StarZMessageBox.ShowDialog($"Current theme exported successfully as backup to {backupFilePath}.", "Success !", false);
                    }
                    LogsManager.Log($"Current theme exported successfully as backup to {backupFilePath}.", logFileName);
                }
            }
            catch (Exception ex)
            {
                HandleExportError(ex);
            }
        }

        private static void HandleExportError(Exception ex)
        {
            LogsManager.Log($"Error exporting current theme: {ex.Message}", logFileName);
            StarZMessageBox.ShowDialog($"Error exporting current theme: {ex.Message}", "Error !", false);
        }

        public static void ImportTheme()
        {
            try
            {
                using OpenFileDialog openFileDialog = new();
                openFileDialog.Filter = "StarZ Theme files (*.szt)|*.szt";
                openFileDialog.Title = "Select a valid StarZ Theme .szt file to import";

                if (openFileDialog.ShowDialog() != DialogResult.OK)
                {
                    LogsManager.Log("Import operation cancelled by user.", logFileName);
                    return;
                }

                ExportTheme(false);

                var importedThemeJson = File.ReadAllText(openFileDialog.FileName);
                var importedThemes = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(importedThemeJson);

                if (importedThemes != null)
                {
                    File.WriteAllText(themeFilePath, importedThemeJson);
                    LogsManager.Log($"Theme imported successfully from {openFileDialog.FileName} and applied.", logFileName);
                    ApplyTheme("CustomTheme");
                }
                else
                {
                    LogsManager.Log("Imported theme file is invalid or corrupt.", logFileName);
                }
            }
            catch (Exception ex)
            {
                HandleImportError(ex);
            }
        }

        private static void HandleImportError(Exception ex)
        {
            StarZMessageBox.ShowDialog($"Error importing theme: {ex.Message}", "Error !", false);
            LogsManager.Log($"Error importing theme: {ex.Message}", logFileName);
        }
    }
}