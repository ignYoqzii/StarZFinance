using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace StarZFinance.Classes
{
    public static class ModelParametersManager
    {
        private static readonly string FilePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "StarZ Finance", "ModelParameters.json");
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

        private static void InitializeDefaultParameters()
        {
            ModelParameters = new()
            {
                { Model.ARIMA, new() { new(Parameter.Epochs, 100), new(Parameter.BatchSize, 32) } },
                { Model.LSTM, new() { new(Parameter.Epochs, 200), new(Parameter.Dropout, 0.2) } },
                { Model.GRU, new() { new(Parameter.Epochs, 150), new(Parameter.Optimizer, "Adam") } }
            };
        }

        public static void UpdateParameterValue(Model model, Parameter paramName, string newValue)
        {
            if (ModelParameters.TryGetValue(model, out var parameters))
            {
                var parameter = parameters.FirstOrDefault(p => p.Name == paramName);
                if (parameter != null)
                {
                    parameter.SetValueFromString(newValue);
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
    }

    public class ModelParameter
    {
        private static readonly Dictionary<Parameter, string> DefaultDescriptions = new()
        {
            { Parameter.Epochs, "Number of training iterations" }, // translation here
            { Parameter.BatchSize, "Samples per batch" },
            { Parameter.Dropout, "Regularization dropout rate" },
            { Parameter.Optimizer, "Optimization algorithm" }
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
                _ => "string"
            };
        }

        public void SetValueFromString(string value)
        {
            Value = value;
        }

        public object GetTypedValue()
        {
            return ValueType! switch
            {
                "int" => int.TryParse(Value!, out var intVal) ? intVal : 0,
                "double" => double.TryParse(Value!, out var doubleVal) ? doubleVal : 0.0,
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
        Dropout,
        Optimizer
    }
}