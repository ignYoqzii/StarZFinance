using StarZFinance.Classes;
using StarZFinance.Windows;
using System.Windows;
using Controls = System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Controls;
using System.Windows.Media.Animation;
using System.Windows.Media;

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

        // Manages the other single settings ScrollViewer
        private void SettingControl_Loaded(object sender, RoutedEventArgs e)
        {
            var SettingControl = sender as ContentControl;

            if (SettingControl!.DataContext is not Setting setting) return;

            if (setting.Type == "Button")
            {
                var button = new Controls.Button
                {
                    Content = setting.DefaultValue,
                    Height = 25,
                    Width = 50,
                    FontSize = 10
                };
                button.Click += (s, args) => setting.Action?.Invoke(null!);
                button.Style = (Style)this.FindResource("DefaultButtons");
                SettingControl.Content = button;
            }
            else if (setting.Type == "CheckBox")
            {
                var checkBox = new Controls.CheckBox
                {
                    IsChecked = (bool)setting.DefaultValue!
                };

                checkBox.Checked += (s, args) =>
                {
                    App.CheckBoxAnimation(s);
                    setting.Action?.Invoke(true);
                };

                checkBox.Unchecked += (s, args) =>
                {
                    App.CheckBoxAnimation(s);
                    setting.Action?.Invoke(false);
                };

                checkBox.Style = (Style)this.FindResource("DefaultCheckBoxes");
                SettingControl.Content = checkBox;
            }
            else if (setting.Type == "TextBox")
            {
                var textBox = new Controls.TextBox
                {
                    Text = (string)setting.DefaultValue!,
                    Height = 25,
                    Width = 200,
                    VerticalAlignment = VerticalAlignment.Center,
                    TextAlignment = TextAlignment.Center
                };
                textBox.TextChanged += (s, args) => setting.Action?.Invoke(textBox.Text);
                textBox.Style = (Style)this.FindResource("DefaultTextBoxes");
                SettingControl.Content = textBox;
            }
        }

        /// <summary>
        /// For the Themes and "Search Setting" functionality
        /// </summary>

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

                    StarZMessageBox.ShowDialog(message, "Error !", false);
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

        private void SearchSettingTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (string.IsNullOrEmpty(SearchSettingTextBox.Text))
            {
                SearchSettingTextBlock.Visibility = Visibility.Visible;
                DisplayAllSettings();
            }
            else
            {
                SearchSettingTextBlock.Visibility = Visibility.Collapsed;
                FilterSettings(SearchSettingTextBox.Text);
            }
        }

        private void DisplayAllSettings()
        {
            AppSettingsItemsControl.ItemsSource = SettingsManager.AppSettings;
        }

        private void FilterSettings(string filterText)
        {
            var filteredSettings = SettingsManager.AppSettings!.Where(setting => setting.Name!.Contains(filterText, StringComparison.OrdinalIgnoreCase)).ToList();
            AppSettingsItemsControl.ItemsSource = filteredSettings;
        }
    }
}
