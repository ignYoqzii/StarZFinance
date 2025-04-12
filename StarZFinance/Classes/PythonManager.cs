using StarZFinance.Windows;
using System.Diagnostics;
using System.IO;
using System.Windows.Controls;

namespace StarZFinance.Classes
{
    public static class PythonManager
    {
        // Get the current application's base directory
        private static readonly string appBaseDirectory = AppDomain.CurrentDomain.BaseDirectory;

        // Path to the Scripts folder inside the base directory
        private static readonly string scriptsDirectory = Path.Combine(appBaseDirectory, "Scripts");

        // Path to the Python executable
        private static readonly string pythonExe = Path.Combine(appBaseDirectory, "EmbeddedPython", "python.exe");

        public static void PredictWithARIMA()
        {
        }

        public static string? PredictWithLSTM(string ticker)
        {
            try
            {
                // Path to the LSTMModel.py script
                string scriptPath = Path.Combine(scriptsDirectory, "LSTMModel.py");

                // Ensure the script file exists
                if (!File.Exists(scriptPath))
                {
                    StarZMessageBox.ShowDialog($"Script not found: {scriptPath}", "Error!", false);
                    return null;
                }

                // Construct the arguments to pass to the Python script
                string arguments = $"\"{scriptPath}\" \"{ticker}\"";  // Passing ticker as argument

                // Set up the process start info
                ProcessStartInfo psi = new()
                {
                    FileName = pythonExe,
                    Arguments = arguments,
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                // Start the process and read the output
                using Process? process = Process.Start(psi);
                string output = process!.StandardOutput.ReadToEnd();
                process.WaitForExit();

                return output.Trim();
            }
            catch (Exception ex)
            {
                StarZMessageBox.ShowDialog($"An error occurred: {ex.Message}", "Error!", false);
                return null;
            }
        }

        public static void PredictWithGRU()
        {
        }
    }
}