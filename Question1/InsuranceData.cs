using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Question1
{
    public class InsuranceData
    {
        [LoadColumn(0)]
        public float Age { get; set; }

        [LoadColumn(1)]
        public string Sex { get; set; }

        [LoadColumn(2)]
        public float Bmi { get; set; }

        [LoadColumn(3)]
        public float Children { get; set; }

        [LoadColumn(4)]
        public string Smoker { get; set; }

        [LoadColumn(5)]
        public string Region { get; set; }

        [LoadColumn(6)]
        public float Charges { get; set; }
    }

    public class InsurancePrediction
    {
        [ColumnName("Score")]
        public float Charges { get; set; }
    }

}
