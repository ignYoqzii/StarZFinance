using Python.Runtime;
using System.IO;
using System.Windows.Controls;

namespace StarZFinance.Classes
{
    public static class PythonManager
    {
        static PythonManager()
        {
            // Initialize Python when the application starts
            InitializePython();
        }

        private static void InitializePython()
        {
            // Determine the path to the embedded Python DLL
            string pythonDllPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "EmbeddedPython", "python312.dll");
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", pythonDllPath);

            // Set the PYTHONHOME environment variable to the embedded Python directory
            string pythonHome = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "EmbeddedPython");
            Environment.SetEnvironmentVariable("PYTHONHOME", pythonHome);

            // Add the embedded Python directory to the PATH environment variable
            string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            Environment.SetEnvironmentVariable("PATH", pythonHome + ";" + path);

            // Initialize the Python engine
            PythonEngine.Initialize();
        }

        public static void PredictWithARIMA()
        {
        }

        public static byte[] PredictWithLSTM(string selectedTicker)
        {
            // Get the parameters from the ModelParametersManager
            int timestep = (int)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.TimeStep)!;
            string feature = ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.Feature)?.ToString()!;
            int lstmUnits = (int)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.LSTMUnits)!;
            double dropoutRate = (double)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.DropoutRate)!;
            int epochs = (int)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.Epochs)!;
            int batchSize = (int)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.BatchSize)!;
            string optimizer = ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.Optimizer)?.ToString()!;
            bool earlyStopping = (bool)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.UseEarlyStopping)!;
            bool showFutureActual = (bool)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.ShowFutureActual)!;
            bool useSentimentAnalysis = (bool)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.UseSentimentAnalysis)!;
            double scalingFactor = (double)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.ScalingFactor)!;
            int plotWindow = (int)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.PlotWindow)!;
            string ticker = selectedTicker;
            string startDate = ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.StartDate)?.ToString()!;
            string endDate = ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.EndDate)?.ToString()!;
            int daysToPredict = (int)ModelParametersManager.GetParameterValue(Model.LSTM, Parameter.DaysToPredict)!;

            using (Py.GIL())
            {
                dynamic sys = Py.Import("sys");
                string scriptFolderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts");
                sys.path.append(scriptFolderPath);

                dynamic lstmModule = Py.Import("LSTMModel");
                dynamic predictor = lstmModule.StockPredictorLSTM(
                    timestep,
                    feature,
                    lstmUnits,
                    dropoutRate,
                    epochs,
                    batchSize,
                    optimizer,
                    earlyStopping,
                    showFutureActual,
                    useSentimentAnalysis,
                    scalingFactor,
                    plotWindow
                    );
                dynamic plotBuffer = predictor.run(ticker, startDate, endDate, daysToPredict);

                // Convert the Python BytesIO object to a C# byte array
                byte[] buffer = plotBuffer.tobytes();
                return buffer;
            }
        }

        public static void PredictWithGRU()
        {
        }

        public static void ShutdownPython()
        {
            // Shutdown the Python engine when the application closes
            PythonEngine.Shutdown();
        }
    }
}