using StarZFinance.Classes;
using System.Windows;
using System.Windows.Controls;

namespace StarZFinance.Pages
{
    /// <summary>
    /// Interaction logic for Home.xaml
    /// </summary>
    public partial class Home : Page
    {
        public Home()
        {
            InitializeComponent();
        }

        private void RunPythonTest_Click(object sender, RoutedEventArgs e)
        {
            PythonManager.ExecutePythonCode(TestPython);
        }
    }
}
