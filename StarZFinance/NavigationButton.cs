using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace StarZFinance
{
    public class NavigationButton : ListBoxItem
    {
        static NavigationButton()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(NavigationButton), new FrameworkPropertyMetadata(typeof(NavigationButton)));
        }

        public Uri NavigationLink
        {
            get { return (Uri)GetValue(NavigationLinkProperty); }
            set { SetValue(NavigationLinkProperty, value); }
        }
        public static readonly DependencyProperty NavigationLinkProperty = DependencyProperty.Register("NavigationLink", typeof(Uri), typeof(NavigationButton), new PropertyMetadata(null));

        public ImageSource Icon
        {
            get { return (ImageSource)GetValue(IconProperty); }
            set { SetValue(IconProperty, value); }
        }
        public static readonly DependencyProperty IconProperty = DependencyProperty.Register("Icon", typeof(ImageSource), typeof(NavigationButton), new PropertyMetadata(null));

    }
}
