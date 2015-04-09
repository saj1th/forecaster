name := "forecaster"

version := "1.0"

scalaVersion := "2.11.6"


libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "1.3.0",
  "org.apache.spark" %% "spark-mllib" % "1.3.0")
