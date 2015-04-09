
package org.ylabs.forecaster


import scopt.OptionParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}



object Predict {

  object RegType extends Enumeration {
    type RegType = Value
    val NONE, L1, L2 = Value
  }

  import RegType._

  case class Params(
                     input: String = null,
                     numIterations: Int = 100,
                     stepSize: Double = 1.0,
                     regType: RegType = L2,
                     regParam: Double = 0.01)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("Forecaster") {
      opt[Int]("numIterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))

      opt[Double]("stepSize")
        .text(s"initial step size, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))

      arg[String]("<input>")
        .required()
        .text("path to train data file")
        .action((x, c) => c.copy(input = x))

      note(
        """
          |e.g.
          |
          | bin/spark-submit --class org.ylabs.forecaster.Predict \
          |  /path/to/forecaster.1.0.jar \
          |  /path/to/csv/data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"Forecaster with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)


  }
}
