import sys
from pyspark import SparkContext, SparkConf
import time
import shutil
def map_to_range(number):
    range_width = 0.1
    range_start = int(number / range_width) * range_width
    range_end = range_start + range_width
    return f"[{range_start}, {range_end})"

if __name__ == "__main__":
    shutil.rmtree("/home/liuwt/output/")
    start_time = time.time()  # 记录开始时间
    conf = SparkConf().setAppName("PySpark Range Mapping Example")
    sc = SparkContext(conf=conf)

    numbers = sc.textFile("/home/liuwt/scores_all_data.txt").flatMap(lambda line: line.split(" "))

    ranges = numbers.map(lambda number: (map_to_range(float(number)), 1))

    range_counts = ranges.reduceByKey(lambda a, b: a + b)
    range_counts = range_counts.coalesce(1)

    end_time = time.time()
    execution_time = end_time - start_time
    range_counts.saveAsTextFile("/home/liuwt/output/")
    print("Execution Time:", execution_time, "seconds")
    
