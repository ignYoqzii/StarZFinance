using StarZFinance.Classes;
using System.Windows;

namespace StarZFinance.Windows
{
    public partial class EditWindow
    {
        public string? NewName { get; private set; }

        public EditWindow()
        {
            InitializeComponent();
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            OverlayService.HideOverlay();
            base.OnClosing(e);
        }

        public static string? ShowDialog(string currentName)
        {
            OverlayService.ShowOverlay();
            EditWindow editWindow = new();
            editWindow.CurrentNameTextBlock.Text = currentName;

            if (editWindow.ShowDialog() == true)
            {
                return editWindow.NewName;
            }
            return null;
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