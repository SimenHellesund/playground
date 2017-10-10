#!/bin/sh                                                                                                                              

CURRENT=1507191500000 #epoch milliseconds                                                                                              
HOURS=5 #interval in hours (batch size)
PERIODS=15 #how many intervals should be run over
OFFSET=`expr $HOURS \* 60 \* 60 \* 1000` #Convert $HOURS to milliseconds
TOTAL=`expr $HOURS \* $PERIODS`
RUNNING=0
TIMESTAMP=$CURRENT

while [ $RUNNING -le $TOTAL ]; do
#debugging output
#echo $TIMESTAMP

python downloading_batching.py $TIMESTAMP $HOURS

TIMESTAMP=`expr $TIMESTAMP - $OFFSET`
RUNNING=`expr $RUNNING + $HOURS`
done


