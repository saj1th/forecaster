package org.bitbucket.saj1th.forecaster

import java.util.Date
import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Success, Try}

import com.datastax.spark.connector.{SomeColumns, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.joda.time.LocalDate
import scopt.OptionParser


/**
 * Enum for regularization params
 */
object RegularizationType extends Enumeration {
  type RegularizationType = Value
  val NONE, L1, L2 = Value
}
import RegularizationType._

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
                   predEnd: String = "2015-12-31",
                   modelSavePath: String = "./",
                   sparkExecutor: String = "",
                   cassandraHost: String = "127.0.0.1")

/**
 * An  app for forecasting sale and volume
 * please use spark-submit to submit the app.
 */
object Forecast extends App with Logging {

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
    opt[String]("sparkexecutor")
      .text("spark executor uri")
      .required()
      .action((x, c) => c.copy(sparkExecutor = x))
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
        | bin/spark-submit
        |  --class org.bitbucket.saj1th.forecaster.Forecast \
        |  /opt/forecaster/bin/forecaster-0.2.jar \
        |  --sparkexecutor hdfs://x.x.x.x:8020/opt/spark-1.2.0-bin-hadoop2.4.tgz
        |  --master mesos://x.x.x.x:5050 \
        |  --cassandrahost x.x.x.x  \
        |  --modelspath  hdfs://path/to/save/models \
        |  hdfs://path/to/csv/data.txt
      """.stripMargin)
  }

  parser.parse(args, defaultParams).map { params =>
    run(params)
  } getOrElse {
    sys.exit(1)
  }


  /**
   * Runs the algorithm
   */
  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName(s"Forecaster with $params")
      .setMaster(params.master)
      .set("spark.executor.memory", "1g")
      .set("spark.executor.uri", params.sparkExecutor)
      .set("spark.hadoop.validateOutputSpecs", "false") //TODO: handle model overwrites elegantly
      .set("spark.cassandra.connection.host", params.cassandraHost)
      //check spark.cassandra.* for additional parameters if needed

    logInfo("running forecaster with" + params)

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

    logInfo("training models")
    for (sku <- ProductData.skus) {
      trainVolumeModel(sku, dailyVolume, algorithm) match {
        case (Success((volModel, scalerModel))) =>
          volumeModels += (sku -> volModel)
          scalerModels += (sku -> scalerModel)

          //scalerModel needs to be calculated only once
          trainSaleModel(sku, dailySale, algorithm, scalerModel) match {
            case (Success((saleModel))) =>
              saleModels += (sku -> saleModel)
            case Failure(ex) =>
              logError("Failed to train sale model for sku:" + sku)
              logError(s"${ex.getMessage}")
          }
        case Failure(ex) =>
          logError("Failed to train volume model for sku:" + sku)
          logError(s"${ex.getMessage}")
      }
    }

    // Do prediction
    logInfo("do prediction")
    val predictions = predictFuture(volumeModels, saleModels, scalerModels, params.predStart, params.predEnd)

    // Save to Cassandra
    logInfo("saving predictions to cassandra")
    val predictionsRdd = sc.parallelize(predictions)
    predictionsRdd.saveToCassandra("forecaster", "predictions", SomeColumns("sku", "date", "sale", "volume"))

    //Save models to HDFS
    logInfo("saving models to HDFS")
    persistModels(volumeModels, saleModels, scalerModels, sc, params.modelSavePath)

    logInfo("done!")
    sc.stop()
  }

  // Trains the Volume Model
  def trainVolumeModel(sku: String,
                       volumes: RDD[Volume],
                       algorithm: LinearRegressionWithSGD): Try[(LinearRegressionModel, StandardScalerModel)] = {
    try {
      //Create labeled data
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
      Success(model, scaler)
    } catch {
      case e: Exception => Failure(e)
    }
  }

  // Trains the Sale Model
  def trainSaleModel(sku: String,
                     sales: RDD[Sale],
                     algorithm: LinearRegressionWithSGD,
                     scaler: StandardScalerModel): Try[LinearRegressionModel] = {

    try {
      //Create labeled data
      val labelData = sales
        .filter(row => row.sku == sku)
        .map { row => LabeledPoint(row.sale, Vectors.dense(row.year, row.month, row.day)) }
      //Feature scaling
      val scaledData = labelData
        .map { data => LabeledPoint(data.label, scaler.transform(Vectors.dense(data.features.toArray))) }
        .cache()

      //Train the algo
      val model = algorithm.run(scaledData)
      Success(model)
    } catch {
      case e: Exception => Failure(e)
    }

  }

  // Perform prediction
  def predictFuture(volumeModels: Map[String, LinearRegressionModel],
                    saleModels: Map[String, LinearRegressionModel],
                    scalerModels: Map[String, StandardScalerModel],
                    predStart: String,
                    predEnd: String): ArrayBuffer[Prediction] = {
    //Get the day iterator
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

  // Iterate over date ranges
  def dayStream(start: LocalDate, end: LocalDate) = Stream.iterate(start)(_ plusDays 1) takeWhile (_ isBefore end)

  //Persist Model Files
  def persistModels(volumeModels: Map[String, LinearRegressionModel],
                    saleModels: Map[String, LinearRegressionModel],
                    scalerModels: Map[String, StandardScalerModel],
                    sc: SparkContext,
                    path: String) = {
    try {
      //Save volume models
      for ((sku, model) <- volumeModels) {
        sc.parallelize(Seq(model), 1).saveAsObjectFile(path + "/volume." + sku + ".model")
      }
      //Save sale models
      for ((sku, model) <- saleModels) {
        sc.parallelize(Seq(model), 1).saveAsObjectFile(path + "/sale." + sku + ".model")
      }
      //Save scaler models
      for ((sku, model) <- scalerModels) {
        sc.parallelize(Seq(model), 1).saveAsObjectFile(path + "/scaler." + sku + ".model")
      }
    } catch {
      case e: Exception => {
        logError("Failed to save models to:" + path)
        logError(s"${e.getMessage}")
      }
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
}

case class Prediction(sku: String, date: Date, sale: Long, volume: Long)
case class Sale(sku: String, year: Int, month: Int, day: Int, sale: Float)
case class Volume(sku: String, year: Int, month: Int, day: Int, volume: Int)
