package org.ylabs.forecaster

import java.util.Date
import scala.collection.mutable.ArrayBuffer

import com.datastax.spark.connector.{SomeColumns, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.LocalDate
import scopt.OptionParser


/**
 * Enum for regularization params
 */
object RegularizationType extends Enumeration {
  type RegularizationType = Value
  val NONE, L1, L2 = Value
}
import org.ylabs.forecaster.RegularizationType._

/**
 * command line params
 */
case class Params(
                   data: String = null,
                   numIterations: Int = 100,
                   stepSize: Double = 1,
                   regType: RegularizationType = L2,
                   regParam: Double = 0.01,
                   master: String = "local[2]",
                   predStart: String = "2015-01-01",
                   predEnd: String = "2015-12-01",
                   modelSavePath: String = "./",
                   cassandraHost: String = "127.0.0.1")

/**
 * An  app for forecasting sale and volume
 * please use spark-submit to submit the app.
 */
object Forecast {

  def main(args: Array[String]) {
    val defaultParams = Params()

    //Parse command line arguments
    val parser = new OptionParser[Params]("Forecaster") {
      opt[String]("master")
        .text("master details")
        .action((x, c) => c.copy(master = x))
      opt[Int]("iterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("stepsize")
        .text(s"initial step size, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[String]("cassandrahost")
        .text("cassandra host")
        .required()
        .action((x, c) => c.copy(cassandraHost = x))
      opt[String]("modelspath")
        .text("path to save models to")
        .required()
        .action((x, c) => c.copy(modelSavePath = x))
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
          |  --cassandraHost host.name.or.ip  \
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
      .setMaster(params.master)
      .set("spark.executor.memory", "1g")
      .set("spark.cassandra.connection.host", params.cassandraHost)

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

    //Regularization Type
    val updater = params.regType match {
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
    var scalerModels: Map[String, StandardScalerModel] = Map()
    for (sku <- ProductData.skus) {
      trainVolumeModel(sku, dailyVolume, algorithm) match {
        case (volModel, scalerModel) =>
          volumeModels += (sku -> volModel)
          scalerModels += (sku -> scalerModel)
          //scalerModel needs to be calculated only once
          saleModels += (sku -> trainSaleModel(sku, dailySale, algorithm, scalerModel))
      }
    }

    // Do prediction
    val predictions = predictFuture(volumeModels, saleModels, scalerModels, params.predStart, params.predEnd)

    // Save to Cassandra
    val predictionsRdd = sc.parallelize(predictions)
    predictionsRdd.saveToCassandra("forecaster", "predictions", SomeColumns("sku", "date", "sale", "volume"))

    //Save models to HDFS
    persistModels(volumeModels, saleModels, scalerModels, sc, params.modelSavePath)

    sc.stop()
  }

  // Trains the Volume Model
  def trainVolumeModel(sku: String,
                       volumes: RDD[Volume],
                       algorithm: LinearRegressionWithSGD): (LinearRegressionModel, StandardScalerModel) = {
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
    (model, scaler)
  }

  // Trains the Sale Model
  def trainSaleModel(sku: String,
                     sales: RDD[Sale],
                     algorithm: LinearRegressionWithSGD,
                     scaler: StandardScalerModel): LinearRegressionModel = {
    val labelData = sales
      .filter(row => row.sku == sku)
      .map { row => LabeledPoint(row.sale, Vectors.dense(row.year, row.month, row.day)) }
    //Feature scaling
    val scaledData = labelData
      .map { data => LabeledPoint(data.label, scaler.transform(Vectors.dense(data.features.toArray))) }
      .cache()

    //Train the algo
    val model = algorithm.run(scaledData)
    model
  }

  // Perform prediction
  def predictFuture(volumeModels: Map[String, LinearRegressionModel],
                    saleModels: Map[String, LinearRegressionModel],
                    scalerModels: Map[String, StandardScalerModel],
                    predStart: String,
                    predEnd: String): ArrayBuffer[Prediction] = {
    val itr = dayStream(new LocalDate(predStart), new LocalDate(predEnd))
    var predictions = new ArrayBuffer[Prediction]()
    for (sku <- ProductData.skus) {
      for (dt <- itr) {
        predictions += Prediction(
          sku,
          dt.toDate,
          volumeModels(sku)
            .predict(scalerModels(sku)
            .transform(Vectors.dense(dt.getYear, dt.getMonthOfYear, dt.getDayOfMonth))).toLong,
          saleModels(sku)
            .predict(scalerModels(sku)
            .transform(Vectors.dense(dt.getYear, dt.getMonthOfYear, dt.getDayOfMonth))).toLong
        )
      }
    }
    predictions
  }

  //Persist Model Files
  def persistModels(volumeModels: Map[String, LinearRegressionModel],
                    saleModels: Map[String, LinearRegressionModel],
                    scalerModels: Map[String, StandardScalerModel],
                    sc: SparkContext,
                    path: String) = {

    for ((sku, model) <- volumeModels) {
      sc.parallelize(Seq(model), 1).saveAsObjectFile(path + "/volume." + sku + ".model")
    }

    for ((sku, model) <- saleModels) {
      sc.parallelize(Seq(model), 1).saveAsObjectFile(path + "/sale." + sku + ".model")
    }

    for ((sku, model) <- scalerModels) {
      sc.parallelize(Seq(model), 1).saveAsObjectFile(path + "/scaler." + sku + ".model")
    }
  }

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
    println("Model: " + model.weights)
    println("Mean Square Error: " + MSE)
  }

  // Iterate over date ranges
  def dayStream(start: LocalDate, end: LocalDate) = Stream.iterate(start)(_ plusDays 1) takeWhile (_ isBefore end)
}

case class Sale(sku: String, year: Int, month: Int, day: Int, sale: Float)

case class Prediction(sku: String, date: Date, sale: Long, volume: Long)

case class Volume(sku: String, year: Int, month: Int, day: Int, volume: Int)
