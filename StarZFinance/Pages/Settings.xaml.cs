using StarZFinance.Classes;
using StarZFinance.Windows;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace StarZFinance.Pages
{
    public partial class Settings : Page
    {
        public static Settings? Instance { get; private set; }

        public Settings()
        {
            InitializeComponent();
            Instance = this;
            string theme = ConfigManager.GetTheme();
            UpdateCheckBoxes(theme); // Must be done when this page loads for the first time (not in App.xaml.cs or ThemesManager.cs)
        }

        public void UpdateCheckBoxes(string theme)
        {
            // Set checkbox states based on the theme
            CheckBoxLightTheme.IsChecked = theme == "LightTheme";
            CheckBoxLightTheme.IsEnabled = theme != "LightTheme";

            CheckBoxDarkTheme.IsChecked = theme == "DarkTheme";
            CheckBoxDarkTheme.IsEnabled = theme != "DarkTheme";

            CheckBoxCustomTheme.IsChecked = theme == "CustomTheme";
            CheckBoxCustomTheme.IsEnabled = theme != "CustomTheme";
        }

        private void CheckBoxTheme_Checked(object sender, RoutedEventArgs e)
        {
            if (sender is System.Windows.Controls.CheckBox checkBox)
            {
                string theme = checkBox.Tag.ToString()!; // Use Tag to determine the theme

                // Load the theme colors
                var colors = ThemesManager.LoadThemeColors(theme);

                // Validate color values using ThemesManager
                if (colors == null || !ThemesManager.AreColorValuesValid(colors))
                {
                    string message = theme == "CustomTheme" ? "Please create a custom theme using the Colors Manager before applying it." : "Invalid color values for the selected theme.";

                    StarZMessageBox.ShowDialog(message, "Error!", false);
                    checkBox.IsChecked = false;
                    return;
                }

                ThemesManager.ApplyTheme(theme);
            }
        }

        private void OpenThemesManager_Click(object sender, RoutedEventArgs e)
        {
            // Create an instance of the color dialog
            StarZColorDialog colorDialog = new();
            OverlayService.ShowOverlay();

            // Show the dialog and wait for user response
            if (colorDialog.ShowDialog() == true)
            {
                // Apply the theme
                string theme = "CustomTheme";
                ThemesManager.ApplyTheme(theme);
            }
            OverlayService.HideOverlay();
        }

        private void ResetAllThemes_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            ThemesManager.ResetThemesToDefault();
        }

        private void ExportThemes_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            ThemesManager.ExportTheme(true);
        }

        private void ImportThemes_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            ThemesManager.ImportTheme();
        }
    }
}
