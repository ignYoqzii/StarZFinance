using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace StarZFinance.Classes
{
    public static class ModelParametersManager
    {
        private static readonly string FilePath = Path.Combine(App.StarZFinanceDirectory, "ModelParameters.json");
        private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

        public static Dictionary<Model, ObservableCollection<ModelParameter>> ModelParameters { get; set; } = [];

        static ModelParametersManager()
        {
            if (File.Exists(FilePath))
                LoadParametersFromFile();
            else
            {
                InitializeDefaultParameters();
                SaveParametersToFile();
            }
        }

        public static void InitializeDefaultParameters()
        {
            ModelParameters = new()
            {
                { Model.ARIMA, new() {
                    new(Parameter.Epochs, 100),
                    new(Parameter.BatchSize, 32)
                } },
                { Model.LSTM, new() {
                    new(Parameter.StartDate, "2010-01-01"),
                    new(Parameter.EndDate, DateTime.Now.ToString("yyyy-MM-dd")),
                    new(Parameter.Features, "Close"),
                    new(Parameter.ShowFutureActual, false),
                    new(Parameter.TimeStep, 60),
                    new(Parameter.LSTMUnits, 50),
                    new(Parameter.DropoutRate, 0.2),
                    new(Parameter.Epochs, 10),
                    new(Parameter.BatchSize, 32),
                    new(Parameter.Optimizer, "adam"),
                    new(Parameter.DaysToPredict, 5),
                    new(Parameter.UseEarlyStopping, false),
                    new(Parameter.UseSentimentAnalysis, false),
                    new(Parameter.ScalingFactor, 5.0)
                } },
                { Model.GRU, new() {
                    new(Parameter.Epochs, 150),
                    new(Parameter.Optimizer, "Adam")
                } }
            };
        }

        public static void UpdateParameterValue(Model model, Parameter paramName, string newValue)
        {
            if (ModelParameters.TryGetValue(model, out var parameters))
            {
                var parameter = parameters.FirstOrDefault(p => p.Name == paramName);
                if (parameter != null)
                {
                    parameter.Value = newValue;
                    SaveParametersToFile();
                }
            }
        }

        public static void SaveParametersToFile() =>
            File.WriteAllText(FilePath, JsonSerializer.Serialize(ModelParameters, JsonOptions));

        public static void LoadParametersFromFile() =>
            ModelParameters = JsonSerializer.Deserialize<Dictionary<Model, ObservableCollection<ModelParameter>>>(File.ReadAllText(FilePath)) ?? [];

        public static object? GetParameterValue(Model model, Parameter paramName)
        {
            if (ModelParameters.TryGetValue(model, out var parameters))
            {
                var parameter = parameters.FirstOrDefault(p => p.Name == paramName);
                return parameter?.GetTypedValue(); // Convert only when retrieving
            }
            return null;
        }

        public static bool IsValueNullOrEmpty(object value)
        {
            return value switch
            {
                null => true,
                string str => string.IsNullOrEmpty(str),
                double d => double.IsNaN(d),
                bool b => !b,
                int i => i == 0,
                _ => false
            };
        }

        public static bool IsValidParameterValue(ModelParameter parameter)
        {
            return parameter.ValueType switch
            {
                "string" => !string.IsNullOrEmpty(parameter.Value as string),
                "double" => double.TryParse(parameter.Value?.ToString(), out _),
                "int" => int.TryParse(parameter.Value?.ToString(), out _),
                "bool" => bool.TryParse(parameter.Value?.ToString(), out _),
                _ => false
            };
        }
    }

    public class ModelParameter
    {
        private static readonly Dictionary<Parameter, string> DefaultDescriptions = new()
        {
            { Parameter.Epochs, "The number of epochs to train the model." }, // translation here
            { Parameter.BatchSize, "The batch size for training." },
            { Parameter.DropoutRate, "The dropout rate to prevent overfitting." },
            { Parameter.Optimizer, "The optimizer to use for compiling the model." },
            { Parameter.StartDate, "The start date for fetching historical stock data (format: \"YYYY-MM-DD\")." },
            { Parameter.EndDate, "The end date for fetching historical stock data (format: \"YYYY-MM-DD\")." },
            { Parameter.Features, "The list of column names to use as features, seperated by commas." },
            { Parameter.ShowFutureActual, "Whether to show actual future values with the predictions to evaluate model accuracy." },
            { Parameter.TimeStep, "The number of previous time steps to consider for predicting the next value." },
            { Parameter.LSTMUnits, "The number of units (neurons) in each LSTM layer." },
            { Parameter.DaysToPredict, "The number of days to predict." },
            { Parameter.UseEarlyStopping, "Whether to use early stopping to prevent overfitting." },
            { Parameter.UseSentimentAnalysis, "Whether to adjust predictions based on sentiment analysis." },
            { Parameter.ScalingFactor, "The factor by which to scale the sentiment adjustment." }
        };

        [JsonPropertyName("Name")]
        public Parameter Name { get; }

        [JsonPropertyName("Description")]
        public string Description { get; }

        [JsonPropertyName("ValueType")]
        public string? ValueType { get; private set; }

        [JsonPropertyName("Value")]
        public string? Value { get; set; }

        [JsonConstructor]
        public ModelParameter(Parameter name, string description, string valueType, string value)
        {
            Name = name;
            Description = description;
            ValueType = valueType;
            Value = value;
        }

        public ModelParameter(Parameter name, object value)
        {
            Name = name;
            Description = DefaultDescriptions.GetValueOrDefault(name, "Custom Parameter");
            SetValue(value);
        }

        public void SetValue(object value)
        {
            Value = value.ToString() ?? string.Empty;
            ValueType = value switch
            {
                int => "int",
                double => "double",
                bool => "bool",
                _ => "string"
            };
        }

        public object GetTypedValue()
        {
            return ValueType! switch
            {
                "int" => int.TryParse(Value!, out var intVal) ? intVal : 0,
                "double" => double.TryParse(Value!, out var doubleVal) ? doubleVal : 0.0,
                "bool" => bool.TryParse(Value!, out var boolVal) && boolVal,
                _ => Value!
            };
        }
    }

    public enum Model
    {
        ARIMA,
        LSTM,
        GRU
    }

    public enum Parameter
    {
        Epochs,
        BatchSize,
        DropoutRate,
        Optimizer,
        StartDate,
        EndDate,
        Features,
        ShowFutureActual,
        TimeStep,
        LSTMUnits,
        DaysToPredict,
        UseEarlyStopping,
        UseSentimentAnalysis,
        ScalingFactor
    }
}
