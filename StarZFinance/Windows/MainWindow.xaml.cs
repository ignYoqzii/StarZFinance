using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Animation;

namespace StarZFinance.Windows
{
    public partial class MainWindow : Window
    {
        public static MainWindow? Instance { get; private set; }

        public MainWindow()
        {
            InitializeComponent();
            Instance = this;
            SideBar.SelectedIndex = 0; // Home tab is shown at first
        }

        // Animation on program's launch, from StarZ Launcher by yoqzii
        private bool isFirstTimeOpened = true; // Only show one time

        private async void Window_Loaded(object sender, RoutedEventArgs e)
        {
            if (isFirstTimeOpened)
            {
                OpeningAnimation.Navigate(new Uri("/Pages/OpeningAnimation.xaml", UriKind.Relative));
                DoubleAnimation animation = new(0, 1, new Duration(TimeSpan.FromSeconds(1)))
                {
                    EasingFunction = new CircleEase() { EasingMode = EasingMode.EaseOut }
                };
                BeginAnimation(OpacityProperty, animation);
                await Task.Delay(1000);
                DoubleAnimation opacityAnimation = new()
                {
                    From = 1,
                    To = 0,
                    Duration = new Duration(TimeSpan.FromSeconds(0.5)),
                };
                OpeningAnimation.BeginAnimation(OpacityProperty, opacityAnimation);
                await Task.Delay(500);
                OpeningAnimation.Visibility = Visibility.Collapsed;
                OpeningAnimation = null;
                isFirstTimeOpened = false;
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Application.Current.Shutdown();
        }

        private double restoreWidth;
        private double restoreHeight;
        private double restoreTop;
        private double restoreLeft;
        private bool isMaximized = false;

        private void MaximizeButton_Click(object sender, RoutedEventArgs e)
        {
            if (!isMaximized)
            {
                MaximizeWindow();
            }
            else
            {
                RestoreWindow();
            }
        }

        private void MaximizeWindow()
        {
            // Store the current size and position
            restoreWidth = this.Width;
            restoreHeight = this.Height;
            restoreTop = this.Top;
            restoreLeft = this.Left;

            // Maximize to the working area
            var workArea = SystemParameters.WorkArea;
            this.Top = workArea.Top;
            this.Left = workArea.Left;
            this.Width = workArea.Width;
            this.Height = workArea.Height;

            isMaximized = true;
        }

        private void RestoreWindow()
        {
            // Restore to original size and position
            this.Width = restoreWidth;
            this.Height = restoreHeight;
            this.Top = restoreTop;
            this.Left = restoreLeft;

            isMaximized = false;
        }

        private void MinimizeButton_Click(object sender, RoutedEventArgs e)
        {
            WindowState = WindowState.Minimized;
        }

        private void TopBar_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            DragMove();
        }

        private void SideBar_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var Selected = SideBar.SelectedItem as NavigationButton;

            NavigationFrame.Navigate(Selected!.NavigationLink);

            PageTextBlock.Text = Selected.Tag.ToString();
        }

        public void ShowOverlay()
        {
            Dispatcher.Invoke(() =>
            {
                WindowPopUpOverlay.Visibility = Visibility.Visible;
            });
        }

        public void HideOverlay()
        {
            Dispatcher.Invoke(() =>
            {
                WindowPopUpOverlay.Visibility = Visibility.Collapsed;
            });
        }
    }
}