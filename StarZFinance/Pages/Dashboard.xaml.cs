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

        private void PredictButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(SelectedTicker))
            {
                StarZMessageBox.ShowDialog("Please enter a valid ticker symbol.", "Error!", false);
                return;
            }
            if (ModelParametersManager.ModelParameters[SelectedModel].Any(p => ModelParametersManager.IsValueNullOrEmpty(p.Value!) || !ModelParametersManager.IsValidParameterValue(p)))
            {
                StarZMessageBox.ShowDialog("Please fill in all the hyperparameters correctly.", "Error!", false);
                return;
            }

            if (SelectedModel == Model.ARIMA)
            {
                int ARIMAepochs = (int)ModelParametersManager.GetParameterValue(SelectedModel, Parameter.Epochs)!;
                // PythonManager.PredictWithARIMA(SelectedTicker, ARIMAepochs);
            }
            else if (SelectedModel == Model.LSTM)
            {
                byte[] plotBuffer = PythonManager.PredictWithLSTM(SelectedTicker);

                using var stream = new System.IO.MemoryStream(plotBuffer);
                var bitmap = new System.Windows.Media.Imaging.BitmapImage();
                bitmap.BeginInit();
                bitmap.StreamSource = stream;
                bitmap.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                bitmap.EndInit();
                PlotZone.Source = bitmap;
            }
            else
            {
                int GRUepochs = (int)ModelParametersManager.GetParameterValue(SelectedModel, Parameter.Epochs)!;
                // PythonManager.PredictWithGRU(SelectedTicker, GRUepochs);
            }
        }
    }
}