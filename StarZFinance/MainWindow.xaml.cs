using System.Windows;
using System.Windows.Media.Animation;
using System.Windows.Controls;
using System.Windows.Input;

namespace StarZFinance
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
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
            Application.Current.Shutdown();
        }

        private void MaximizeButton_Click(object sender, RoutedEventArgs e)
        {
            if (WindowState == WindowState.Normal)
            {
                WindowState = WindowState.Maximized;
            }
            else
            {
                WindowState = WindowState.Normal;
            }
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
    }
}