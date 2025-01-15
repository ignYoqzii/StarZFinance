using  StarZFinance.Windows;

namespace StarZFinance.Classes
{
    public static class OverlayService
    {
        public static void ShowOverlay()
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() =>
            {
                MainWindow.Instance?.ShowOverlay();
            });
        }

        public static void HideOverlay()
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() =>
            {
                MainWindow.Instance?.HideOverlay();
            });
        }
    }
}
