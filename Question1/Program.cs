using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Question1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Create MLContext
            var mlContext = new MLContext();

            // Load data
            string dataPath = "insurance.csv";
            IDataView dataView = mlContext.Data.LoadFromTextFile<InsuranceData>(dataPath, hasHeader: true, separatorChar: ',');

            // Data preparation
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
            {
                new InputOutputColumnPair("Sex", "Sex"),
                new InputOutputColumnPair("Smoker", "Smoker"),
                new InputOutputColumnPair("Region", "Region")
            })
            .Append(mlContext.Transforms.Conversion.ConvertType("Children", outputKind: DataKind.Single))
            .Append(mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Bmi", "Children", "Smoker", "Region"))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Charges", featureColumnName: "Features"));

            // Train model
            var model = pipeline.Fit(dataView);

            // Evaluate model
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Charges");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:               {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:0.##}");
            Console.WriteLine($"**************************************************\n");

            // Save model
            mlContext.Model.Save(model, dataView.Schema, "InsuranceModel.zip");

            // Predict sample
            var sample = new InsuranceData()
            {
                Age = 28,
                Sex = "Female",
                Bmi = 25.5f,
                Children = 1,
                Smoker = "No",
                Region = "Southwest"
            };
            var predictor = mlContext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var prediction = predictor.Predict(sample);
            Console.WriteLine($"**************************************************");
            Console.WriteLine($"*       Sample Prediction         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Age:                          {sample.Age}");
            Console.WriteLine($"*       Sex:                          {sample.Sex}");
            Console.WriteLine($"*       BMI:                          {sample.Bmi}");
            Console.WriteLine($"*       Children:                     {sample.Children}");
            Console.WriteLine($"*       Smoker:                       {sample.Smoker}");
            Console.WriteLine($"*       Region:                       {sample.Region}");
            Console.WriteLine($"*       Predicted Charges:            ${prediction.Charges}");
            Console.WriteLine($"*************************************************");

        }
    }
}
