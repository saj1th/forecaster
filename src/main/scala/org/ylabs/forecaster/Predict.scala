package org.ylabs.forecaster

import org.apache.spark.{HashPartitioner, SparkContext}

object Predict {


  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local[2]", "SalesHistory")


    val data = sc.textFile("/Users/asajith/Dev/data/mockdata.csv").map(_.split(","))

    //hash-partitioning to send the elements with the same hash key across
    //the network to the same machine to reduce shuffle/chatter during join.
    //persist() to cache and prevent rework.
    val dailyVolume = data.map(r => (r(0).concat(":").concat(r(1)), 1))
      .reduceByKey((x, y) => x + y)
      .partitionBy(new HashPartitioner(10))
      .persist()

    //hash-partitioning & persist()
    val dailySale = data.map(r => (r(0).concat(":").concat(r(1)), r(2).toFloat))
      .reduceByKey((x, y) => x + y)
      .partitionBy(new HashPartitioner(10))
      .persist()

    //join to get daily sales/volume aggregation
    val aggregate = dailyVolume.join(dailySale)
      .flatMap { case (x: String, (y: Int, z: Float)) => parse(x, y, z) }

    val splits = aggregate.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

  }


  def parse(x: String, y: Int, z: Float) = {
    val split = x.split(':')
    val dtSplit = split(1).split('-')
    Some(split(0), split(0).hashCode % 50, dtSplit(0).toInt, dtSplit(1).toInt, dtSplit(2).toInt, y, z)
  }
}
