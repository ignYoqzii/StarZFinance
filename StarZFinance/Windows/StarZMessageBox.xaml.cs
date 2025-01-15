using StarZFinance.Classes;
using System;
using System.Windows;
using System.Windows.Input;
using static StarZFinance.Windows.MainWindow;

namespace StarZFinance.Windows
{
    /// <summary>
    /// I didn't like the default MessageBox provided by Windows, so I made my own.
    /// </summary>
    public partial class StarZMessageBox
    {
        public StarZMessageBox()
        {
            InitializeComponent();
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            OverlayService.HideOverlay();
            base.OnClosing(e); // Call the base method
        }

        public static bool? ShowDialog(string message, string title, bool showCancelButton = true)
        {
            OverlayService.ShowOverlay();
            StarZMessageBox messageBox = new();
            messageBox.Message.Text = message;
            messageBox.MessageTitle.Text = title;
            messageBox.CancelButton.Visibility = showCancelButton ? Visibility.Visible : Visibility.Collapsed;
            return messageBox.ShowDialog();
        }


        private void OKButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }

        private void CloseButton_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            DialogResult = false;
        }
    }
}
