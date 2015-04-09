package org.ylabs.forecaster


import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{Updater, L1Updater, SquaredL2Updater, SimpleUpdater}
import org.apache.spark.mllib.regression.{LinearRegressionModel, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser


//Enum for regularization params
object RegzationType extends Enumeration {
  type RegzationType = Value
  val NONE, L1, L2 = Value
}
import RegzationType._

/**
 * An  app for forecasting sale and volume
 * please use spark-submit to submit the app.
 */
object Forecast {

  def main(args: Array[String]) {
    val defaultParams = Params()

    //Parse command line arguments
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
        .action((x, c) => c.copy(data = x))
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

  /**
   * Runs the algorithm
   */
  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"Forecaster with $params")
      .setMaster("local[2]")
      .set("spark.executor.memory", "1g")

    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    //Load data from file
    val data = sc.textFile(params.data).map(_.split(","))

    // Aggregate number of sales per day per product
    // count over GROUP BY (sku +':'+date)
    val dailyVolume = data.map(r => (r(0).concat(":").concat(r(1)), 1))
      .reduceByKey((x, y) => x + y)
      .map(r => parseVolume(r._1, r._2))
      .persist()

    // Aggregate sale amount per day per product
    // sum of sales over GROUP BY (sku +':'+date)
    val dailySale = data.map(r => (r(0).concat(":").concat(r(1)), r(2).toFloat))
      .reduceByKey((x, y) => x + y)
      .map(r => parseSale(r._1, r._2))
      .persist()

    //ridge, lasso, or simple regression
    val updater = params.regzationType match {
      case NONE => new SimpleUpdater()
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }
    val algorithm = new LinearRegressionWithSGD()
    algorithm.optimizer
      .setNumIterations(params.numIterations)
      .setStepSize(params.stepSize)
      .setUpdater(updater)
      .setRegParam(params.regParam)


    var volumeModels: Map[String, LinearRegressionModel] = Map()
    var saleModels: Map[String, LinearRegressionModel] = Map()

    for (sku <- ProductData.skus) {
      volumeModels += (sku -> trainVolumeModel(sku, dailyVolume, algorithm))
      saleModels += (sku -> trainSaleModel(sku, dailySale, algorithm))
    }

//    predictAndPersist(volumeModels, saleModels)

    sc.stop()
  }

  def trainVolumeModel(sku: String, volumes: RDD[Volume], algorithm: LinearRegressionWithSGD): LinearRegressionModel = {
    val labelData = volumes
      .filter(row => row.sku == sku)
      .map { row => LabeledPoint(row.volume, Vectors.dense(row.year, row.month, row.day)) }

    //Feature scaling to standardize the range of independent variables
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(labelData.map(x => x.features))
    val scaledData = labelData
      .map { data => LabeledPoint(data.label, scaler.transform(Vectors.dense(data.features.toArray))) }
      .cache()

    //Train the algo
    val model = algorithm.run(scaledData)
    model
  }

  def trainSaleModel(sku: String, sales: RDD[Sale], algorithm: LinearRegressionWithSGD): LinearRegressionModel = {
    val labelData = sales
      .filter(row => row.sku == sku)
      .map { row => LabeledPoint(row.sale, Vectors.dense(row.year, row.month, row.day)) }

    //Feature scaling to standardize the range of independent variables
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(labelData.map(x => x.features))
    val scaledData = labelData
      .map { data => LabeledPoint(data.label, scaler.transform(Vectors.dense(data.features.toArray))) }
      .cache()

    //Train the algo
    val model = algorithm.run(scaledData)
    model
  }

//  def predictAndPersist(volumeModels: LinearRegressionModel, saleModels: LinearRegressionModel) = {
//    for (sku <- ProductData.skus) {
//
//    }
//  }

  def parseVolume(x: String, y: Int) = {
    //split sku and date
    val split = x.split(':')
    //Split to year, month and day
    val dtSplit = split(1).split('-')
    Volume(split(0), dtSplit(0).toInt, dtSplit(1).toInt, dtSplit(2).toInt, y)
  }

  def parseSale(x: String, y: Float) = {
    //split sku and date
    val split = x.split(':')
    //Split to year, month and day
    val dtSplit = split(1).split('-')
    Sale(split(0), dtSplit(0).toInt, dtSplit(1).toInt, dtSplit(2).toInt, y)
  }

  def checkMSE(model: LinearRegressionModel, testData: RDD[LabeledPoint]) = {
    //determine how well the model predicts the test data
    //measures the average of the squares of the "errors"
    val valsAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val power = valsAndPreds.map {
      case (v, p) => math.pow((v - p), 2)
    }

    // Mean Square Error
    val MSE = power.reduce((a, b) => a + b) / power.count()
    println("Mean Square Error: " + MSE)
    println("Model: " + model.weights)
  }

}


//Holds the run params
case class Params(
        data: String = null,
        numIterations: Int = 100,
        stepSize: Double = 1,
        regzationType: RegzationType = L2,
        regParam: Double = 0.01,
        cassandraServer: String = "127.0.0.1")

case class Sale(sku: String, year: Int, month: Int, day: Int, sale: Float)

case class Volume(sku: String, year: Int, month: Int, day: Int, volume: Int)
