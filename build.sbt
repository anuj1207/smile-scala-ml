name := "smile-scala-ml"

version := "1.0"

scalaVersion := "2.12.3"

libraryDependencies ++= Seq(
  "com.github.haifengl" %% "smile-scala" % "1.4.0",
  "org.slf4j" % "slf4j-api" % "1.7.25",
  "ch.qos.logback" % "logback-classic" % "1.1.3" % Runtime
)