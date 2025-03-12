using StarZFinance.Classes;
using StarZFinance.Windows;
using System.Windows;
using System.Windows.Controls;

namespace StarZFinance.Pages
{
    /// <summary>
    /// Interaction logic for Dashboard.xaml
    /// </summary>
    public partial class Dashboard : Page
    {
        public Model SelectedModel;

        public Dashboard()
        {
            InitializeComponent();
            ModelChoiceComboBox.SelectedIndex = 0;
        }

        private void SearchParameterTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (string.IsNullOrEmpty(SearchParameterTextBox.Text))
            {
                SearchParameterTextBlock.Visibility = Visibility.Visible;
                DisplayAllSettings();
            }
            else
            {
                SearchParameterTextBlock.Visibility = Visibility.Collapsed;
                FilterSettings(SearchParameterTextBox.Text);
            }
        }

        private void DisplayAllSettings()
        {
            ModelParametersItemsControl.ItemsSource = ModelParametersManager.ModelParameters[SelectedModel];
        }

        private void FilterSettings(string filterText)
        {
            var filteredSettings = ModelParametersManager.ModelParameters[SelectedModel].Where(parameter => parameter.Name.ToString().Contains(filterText, StringComparison.OrdinalIgnoreCase)).ToList();
            ModelParametersItemsControl.ItemsSource = filteredSettings;
        }

        private void ModelChoiceComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ModelChoiceComboBox.SelectedItem is ComboBoxItem selectedItem && selectedItem.Tag is string tag)
            {
                if (Enum.TryParse(tag, out Model model))
                {
                    SelectedModel = model;
                    ModelParametersItemsControl.ItemsSource = ModelParametersManager.ModelParameters[SelectedModel];
                    ModelParametersTitleTextBlock.Text = $"{selectedItem.Tag} Hyperparameters"; // translation here
                }
            }
        }

        private void ParameterTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (sender is System.Windows.Controls.TextBox textBox && textBox.DataContext is ModelParameter param)
            {
                ModelParametersManager.UpdateParameterValue(SelectedModel, param.Name, textBox.Text);
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            int epoch = (int)ModelParametersManager.GetParameterValue(SelectedModel, Parameter.Epochs)!;
            StarZMessageBox.ShowDialog($"Epochs: {epoch+10}", "Epochs", false);
        }
    }
}