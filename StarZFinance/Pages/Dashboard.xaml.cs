using StarZFinance.Classes;
using StarZFinance.Windows;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Media.Imaging;

namespace StarZFinance.Pages
{
    /// <summary>
    /// Interaction logic for Dashboard.xaml
    /// </summary>
    public partial class Dashboard : Page
    {
        public Model SelectedModel { get; private set; } = Model.ARIMA; // Default model
        public string? SelectedTicker { get; private set; }

        public Dashboard()
        {
            InitializeComponent();
            Initialize();
        }

        private void Initialize()
        {
            if (Enum.TryParse(ConfigManager.GetSelectedModel(), out Model model))
            {
                SelectedModel = model;
            }
            ModelChoiceComboBox.SelectedIndex = (int)SelectedModel;
        }

        private void SearchParameterTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            string searchText = SearchParameterTextBox.Text;
            SearchParameterTextBlock.Visibility = string.IsNullOrEmpty(searchText) ? Visibility.Visible : Visibility.Collapsed;
            ModelParametersItemsControl.ItemsSource = string.IsNullOrEmpty(searchText)
                ? ModelParametersManager.ModelParameters[SelectedModel]
                : ModelParametersManager.ModelParameters[SelectedModel].Where(p => p.Name.ToString().Contains(searchText, StringComparison.OrdinalIgnoreCase));
        }

        private void TickerTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            SelectedTicker = TickerTextBox.Text;
            TickerTextBlock.Visibility = string.IsNullOrEmpty(SelectedTicker) ? Visibility.Visible : Visibility.Collapsed;
        }

        private void ModelChoiceComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e) // Triggered when SelectedIndex or SelectedItem changes
        {
            if (ModelChoiceComboBox.SelectedItem is ComboBoxItem selectedItem &&
                Enum.TryParse(selectedItem.Tag?.ToString(), out Model model))
            {
                SelectedModel = model;
                ConfigManager.SetSelectedModel(selectedItem.Tag.ToString()!);
                ModelParametersItemsControl.ItemsSource = ModelParametersManager.ModelParameters[SelectedModel];
                ModelParametersTitleTextBlock.Text = $"{SelectedModel} Hyperparameters";
            }
        }

        private void ParameterTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (sender is System.Windows.Controls.TextBox textBox && textBox.DataContext is ModelParameter param)
            {
                ModelParametersManager.UpdateParameterValue(SelectedModel, param.Name, textBox.Text);
            }
        }

        private async void TrainAndPredictButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(SelectedTicker))
            {
                StarZMessageBox.ShowDialog("Please enter a valid ticker symbol.", "Error!", false);
                return;
            }

            if (ModelParametersManager.ModelParameters[SelectedModel].Any(p =>
                ModelParametersManager.IsValueNullOrEmpty(p.Value!) ||
                !ModelParametersManager.IsValidParameterValue(p)))
            {
                StarZMessageBox.ShowDialog("Please fill in all the hyperparameters correctly.", "Error!", false);
                return;
            }

            PredictionStatusTextBlock.Text = "Training the model can take several minutes. UI may freeze. Please wait...";
            switch (SelectedModel)
            {
                case Model.ARIMA:
                    // ARIMA logic here
                    break;

                case Model.LSTM:
                    await HandlePredictions("LSTMModel.py");
                    break;

                case Model.GRU:
                    await HandlePredictions("GRUModel.py");
                    break;
            }
            PredictionStatusTextBlock.Text = "";
        }

        private async Task HandlePredictions(string scriptName)
        {
            string? result = await Task.Run(() => PythonManager.Predict(SelectedTicker!, scriptName));
            if (result != null)
            {
                try
                {
                    byte[] imageBytes = Convert.FromBase64String(result);
                    using MemoryStream ms = new(imageBytes);
                    BitmapImage bitmapImage = new();
                    ms.Seek(0, SeekOrigin.Begin);
                    bitmapImage.BeginInit();
                    bitmapImage.StreamSource = ms;
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.EndInit();
                    PlotZone.Source = bitmapImage;
                }
                catch (Exception ex)
                {
                    StarZMessageBox.ShowDialog($"An error occurred while processing the predictions result: {ex.Message}", "Error!", false);
                }
            }
            else
            {
                StarZMessageBox.ShowDialog("An error occurred while running the predictions.", "Error!", false);
            }
        }
    }
}