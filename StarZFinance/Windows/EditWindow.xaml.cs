using System.Windows;
using static StarZFinance.Windows.MainWindow;

namespace StarZFinance.Windows
{
    public partial class EditWindow
    {
        public string? CurrentName { get; private set; }
        public string? NewName { get; private set; }
        public string? CurrentURL { get; private set; }
        public string? NewURL { get; private set; }

        public EditWindow(string currentName, string? currentUrl = null)
        {
            InitializeComponent();
            CurrentName = currentName;
            DataContext = this;
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            // Hide the BackgroundForWindowsOnTop element when closing

            base.OnClosing(e); // Call the base method
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            NewName = NewNameTextBox.Text.Trim();
            DialogResult = true;
        }
    }
}

