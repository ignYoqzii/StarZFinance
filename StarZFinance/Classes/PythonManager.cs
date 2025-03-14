using Python.Runtime;
using System.IO;
using System.Windows.Controls;

namespace StarZFinance.Classes
{
    public static class PythonManager
    {
        public static void ExecutePythonCode(TextBlock textBlock)
        {
            // Determine the path to the embedded Python DLL
            string pythonDllPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "EmbeddedPython", "python313.dll");
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", pythonDllPath);

            // Set the PYTHONHOME environment variable to the embedded Python directory
            string pythonHome = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "EmbeddedPython");
            Environment.SetEnvironmentVariable("PYTHONHOME", pythonHome);

            // Add the embedded Python directory to the PATH environment variable
            string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            Environment.SetEnvironmentVariable("PATH", pythonHome + ";" + path);

            // Initialize the Python engine
            PythonEngine.Initialize();

            using (Py.GIL())
            {
                dynamic sys = Py.Import("sys");

                // Determine the script folder path
                string scriptFolderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts");

                // Add the script folder to sys.path
                sys.path.append(scriptFolderPath);

                dynamic calculatorModule = Py.Import("LSTM");

                // Call the Calculator function
                int result = (int)calculatorModule.Calculator(5, 10);
                textBlock.Text = result.ToString();
            }

            // Shutdown the Python engine
            PythonEngine.Shutdown();
        }
    }
}