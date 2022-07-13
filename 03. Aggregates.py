# Databricks notebook source
# MAGIC %md ## Line Aggregation
# MAGIC
# MAGIC Instead of the point-to-point evaluation, we will instead be aggregating into lines and comparing as such.

# COMMAND ----------

from pyspark.sql.functions import *
import mosaic as mos

spark.conf.set("spark.databricks.labs.mosaic.geometry.api", "ESRI")
spark.conf.set("spark.databricks.labs.mosaic.index.system", "H3")
mos.enable_mosaic(spark, dbutils)

# COMMAND ----------

cargos_indexed = (
    spark.read.table("ship2ship.cargos_indexed")
    .repartition(sc.defaultParallelism * 20)
    .filter(
        col("timestamp").between(
            "2018-01-31T00:00:00.000+0000", "2018-01-31T23:59:00.000+0000"
        )
    )
)
display(cargos_indexed)

# COMMAND ----------

cargos_indexed.count()

# COMMAND ----------

# MAGIC %md ## Create Lines
# MAGIC
# MAGIC We can `groupBy` across a timewindow to give us aggregated geometries to work with.

# COMMAND ----------

lines = (
    cargos_indexed.repartition(sc.defaultParallelism * 20)
    .groupBy("mmsi", window("timestamp", "15 minutes"))
    .agg(collect_list(struct(col("point_geom"), col("timestamp"))).alias("coords"))
    .withColumn(
        "coords",
        expr(
            """
      array_sort(coords, (left, right) -> 
        case 
          when left.timestamp < right.timestamp then -1 
          when left.timestamp > right.timestamp then 1 
          else 0 
        end
      )"""
        ),
    )
    .withColumn("line", mos.st_makeline(col("coords.point_geom")))
    .cache()
)

# COMMAND ----------

lines.count()

# COMMAND ----------

one_metre = 0.00001 - 0.000001
buffer = 200 * one_metre


def get_buffer(line):
    np = expr(f"st_numpoints({line})")
    max_np = lines.select(max(np)).collect()[0][0]
    return (
        lit(max_np) * lit(buffer) / np
    )  # inverse proportional to number of points, larger buffer for slower ships


cargo_movement = (
    lines.withColumn("buffer_r", get_buffer("line"))
    .withColumn("buffer_geom", mos.st_buffer("line", col("buffer_r")))
    .withColumn("buffer", mos.st_astext("buffer_geom"))
    .withColumn("ix", mos.mosaic_explode("buffer_geom", lit(9)))
)

(
    cargo_movement
    # .filter("mmsi == 636016431")
    # .select(col("ix.index_id").alias('ix'))
    .createOrReplaceTempView("ship_path")
)

# COMMAND ----------

display(spark.read.table("ship_path"))

# COMMAND ----------

to_plot = spark.read.table("ship_path").select("buffer").distinct()

# COMMAND ----------

to_plot.count()

# COMMAND ----------

# DBTITLE 1,Example Lines
# MAGIC %%mosaic_kepler
# MAGIC to_plot "buffer" "geometry" 3000

# COMMAND ----------

# MAGIC %md ## Find All Candidates

# COMMAND ----------

candidates_lines = (
    cargo_movement.alias("a")
    .join(
        cargo_movement.alias("b"),
        [
            col("a.ix.index_id") == col("b.ix.index_id"),
            col("a.mmsi") < col("b.mmsi"),
            col("a.window") == col("b.window"),
        ],
    )
    .where(
        (col("a.ix.is_core") | col("b.ix.is_core"))
        | mos.st_intersects("a.ix.wkb", "b.ix.wkb")
    )
    .select(
        col("a.mmsi").alias("vessel_1"),
        col("b.mmsi").alias("vessel_2"),
        col("a.window").alias("window"),
        col("a.buffer").alias("line_1"),
        col("b.buffer").alias("line_2"),
        col("a.ix.index_id").alias("index"),
    )
    .drop_duplicates()
)

(
    candidates_lines.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("ship2ship.overlap_candidates_lines")
)

# 18.73 min without Photon
# 19.51 min with Photon

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ship2ship.overlap_candidates_lines;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW agg_overlap AS
# MAGIC SELECT index AS ix, count(*) AS count, first(line_1) as line_1, first(line_2) as line_2
# MAGIC FROM ship2ship.overlap_candidates_lines
# MAGIC GROUP BY ix
# MAGIC ORDER BY count DESC

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC "agg_overlap" "ix" "h3" 2000

# COMMAND ----------

harbours_h3 = spark.read.table("ship2ship.harbours_h3")
candidates = spark.read.table("ship2ship.overlap_candidates_lines")

# COMMAND ----------

matches = (
    candidates_lines.join(
        harbours_h3, how="leftouter", on=candidates_lines["index"] == harbours_h3["h3"]
    )
    .where(harbours_h3["h3"].isNull())
    .groupBy("vessel_1", "vessel_2")
    .agg(first("line_1").alias("line_1"), first("line_2").alias("line_2"))
)

# COMMAND ----------

matches.count()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC matches "line_1" "geometry" 2000

# COMMAND ----------
