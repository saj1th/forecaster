name := "forecaster"

version := "1.0"

scalaVersion := "2.11.6"

// additional libraries
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.2.0",
  "org.apache.spark" %% "spark-sql" % "1.2.0",
  "org.apache.spark" %% "spark-mllib" % "1.2.0",
  "com.github.scopt" %% "scopt" % "3.3.0",
  "com.github.nscala-time" %% "nscala-time" % "1.8.0",
  "com.datastax.spark" %% "spark-cassandra-connector" % "1.2.0-rc3"
)

resolvers ++= Seq(
  "Cloudera Repository" at "https://repository.cloudera.com/artifactory/cloudera-repos/",
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Apache HBase" at "https://repository.apache.org/content/repositories/releases",
  "Twitter Maven Repo" at "http://maven.twttr.com/",
  "scala-tools" at "https://oss.sonatype.org/content/groups/scala-tools",
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
  "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
  "Mesosphere Public Repository" at "http://downloads.mesosphere.io/maven",
  "Sonatype OSS Repo" at "https://oss.sonatype.org/content/repositories/releases",
  "Sonatype OSS Snapshots Repo" at "http://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype OSS Tools Repo" at "https://oss.sonatype.org/content/groups/scala-tools",
  "Concurrent Maven Repo" at "http://conjars.org/repo"
)
