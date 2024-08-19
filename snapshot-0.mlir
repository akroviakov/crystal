module {
  func.func @main() {
    %0 = relalg.query (){
      %1 = relalg.basetable  {rows = 0x4156E48FC0000000 : f64, table_identifier = "lineorder"} columns: {lo_commitdate => @lineorder::@lo_commitdate({type = !db.nullable<i32>}), lo_custkey => @lineorder::@lo_custkey({type = !db.nullable<i32>}), lo_discount => @lineorder::@lo_discount({type = !db.nullable<i32>}), lo_extendedprice => @lineorder::@lo_extendedprice({type = !db.nullable<!db.decimal<18, 2>>}), lo_linenumber => @lineorder::@lo_linenumber({type = !db.nullable<i32>}), lo_orderdate => @lineorder::@lo_orderdate({type = !db.nullable<i32>}), lo_orderkey => @lineorder::@lo_orderkey({type = !db.nullable<i32>}), lo_orderpriority => @lineorder::@lo_orderpriority({type = !db.nullable<!db.string>}), lo_ordtotalprice => @lineorder::@lo_ordtotalprice({type = !db.nullable<!db.decimal<18, 2>>}), lo_partkey => @lineorder::@lo_partkey({type = !db.nullable<i32>}), lo_quantity => @lineorder::@lo_quantity({type = !db.nullable<i32>}), lo_revenue => @lineorder::@lo_revenue({type = !db.nullable<!db.decimal<18, 2>>}), lo_shipmode => @lineorder::@lo_shipmode({type = !db.nullable<!db.string>}), lo_shippriority => @lineorder::@lo_shippriority({type = !db.nullable<!db.char<1>>}), lo_suppkey => @lineorder::@lo_suppkey({type = !db.nullable<i32>}), lo_supplycost => @lineorder::@lo_supplycost({type = !db.nullable<!db.decimal<18, 2>>}), lo_tax => @lineorder::@lo_tax({type = !db.nullable<i32>})} loc(#loc3)
      %2 = relalg.selection %1 (%arg0: !tuples.tuple){
        %14 = db.constant(19940210 : i32) : i32 loc(#loc5)
        %15 = db.constant(19940204 : i32) : i32 loc(#loc5)
        %16 = tuples.getcol %arg0 @lineorder::@lo_orderdate : !db.nullable<i32> loc(#loc6)
        %17 = db.between %16 : !db.nullable<i32> between %15 : i32, %14 : i32, lowerInclusive : true, upperInclusive : true loc(#loc7)
        tuples.return %17 : !db.nullable<i1> loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 3.000000e+00 : f64, rows = 5860.5615234375 : f64, selectivity = 9.765625E-4 : f64} loc(#loc4)
      %3 = relalg.selection %2 (%arg0: !tuples.tuple){
        %14 = db.constant(7 : i32) : i32 loc(#loc5)
        %15 = db.constant(5 : i32) : i32 loc(#loc5)
        %16 = tuples.getcol %arg0 @lineorder::@lo_discount : !db.nullable<i32> loc(#loc8)
        %17 = db.between %16 : !db.nullable<i32> between %15 : i32, %14 : i32, lowerInclusive : true, upperInclusive : true loc(#loc9)
        tuples.return %17 : !db.nullable<i1> loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 3.000000e+00 : f64, rows = 5860.5615234375 : f64, selectivity = 0.0947265625 : f64} loc(#loc4)
      %4 = relalg.selection %3 (%arg0: !tuples.tuple){
        %14 = db.constant(35 : i32) : i32 loc(#loc5)
        %15 = db.constant(26 : i32) : i32 loc(#loc5)
        %16 = tuples.getcol %arg0 @lineorder::@lo_quantity : !db.nullable<i32> loc(#loc10)
        %17 = db.between %16 : !db.nullable<i32> between %15 : i32, %14 : i32, lowerInclusive : true, upperInclusive : true loc(#loc11)
        tuples.return %17 : !db.nullable<i1> loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 3.000000e+00 : f64, rows = 5860.5615234375 : f64, selectivity = 0.1572265625 : f64} loc(#loc4)
      %5 = relalg.selection %4 (%arg0: !tuples.tuple){
        %14 = tuples.getcol %arg0 @lineorder::@lo_orderdate : !db.nullable<i32> loc(#loc6)
        %15 = db.isnull %14 : <i32> loc(#loc5)
        %16 = db.not %15 : i1 loc(#loc5)
        tuples.return %16 : i1 loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 1.000000e+03 : f64, rows = 5860.5615234375 : f64, selectivity = 1.000000e+00 : f64} loc(#loc4)
      %6 = relalg.selection %5 (%arg0: !tuples.tuple){
        %14 = tuples.getcol %arg0 @lineorder::@lo_discount : !db.nullable<i32> loc(#loc8)
        %15 = db.isnull %14 : <i32> loc(#loc5)
        %16 = db.not %15 : i1 loc(#loc5)
        tuples.return %16 : i1 loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 1.000000e+03 : f64, rows = 5860.5615234375 : f64, selectivity = 1.000000e+00 : f64} loc(#loc4)
      %7 = relalg.selection %6 (%arg0: !tuples.tuple){
        %14 = tuples.getcol %arg0 @lineorder::@lo_quantity : !db.nullable<i32> loc(#loc10)
        %15 = db.isnull %14 : <i32> loc(#loc5)
        %16 = db.not %15 : i1 loc(#loc5)
        tuples.return %16 : i1 loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 1.000000e+03 : f64, rows = 5860.5615234375 : f64, selectivity = 1.000000e+00 : f64} loc(#loc4)
      %8 = relalg.selection %7 (%arg0: !tuples.tuple){
        %14 = tuples.getcol %arg0 @lineorder::@lo_discount : !db.nullable<i32> loc(#loc8)
        %15 = db.isnull %14 : <i32> loc(#loc5)
        %16 = db.not %15 : i1 loc(#loc5)
        tuples.return %16 : i1 loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 1.000000e+03 : f64, rows = 5860.5615234375 : f64, selectivity = 1.000000e+00 : f64} loc(#loc4)
      %9 = relalg.selection %8 (%arg0: !tuples.tuple){
        %14 = tuples.getcol %arg0 @lineorder::@lo_orderdate : !db.nullable<i32> loc(#loc6)
        %15 = db.isnull %14 : <i32> loc(#loc5)
        %16 = db.not %15 : i1 loc(#loc5)
        tuples.return %16 : i1 loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 1.000000e+03 : f64, rows = 5860.5615234375 : f64, selectivity = 1.000000e+00 : f64} loc(#loc4)
      %10 = relalg.selection %9 (%arg0: !tuples.tuple){
        %14 = tuples.getcol %arg0 @lineorder::@lo_quantity : !db.nullable<i32> loc(#loc10)
        %15 = db.isnull %14 : <i32> loc(#loc5)
        %16 = db.not %15 : i1 loc(#loc5)
        tuples.return %16 : i1 loc(#loc4)
      } attributes {cost = 5860.5615234375 : f64, evaluationCost = 1.000000e+03 : f64, rows = 5860.5615234375 : f64, selectivity = 1.000000e+00 : f64} loc(#loc4)
      %11 = relalg.map %10 computes : [@map0::@tmp_attr1({type = !db.nullable<!db.decimal<38, 2>>})] (%arg0: !tuples.tuple){
        %14 = tuples.getcol %arg0 @lineorder::@lo_extendedprice : !db.nullable<!db.decimal<18, 2>> loc(#loc13)
        %15 = tuples.getcol %arg0 @lineorder::@lo_discount : !db.nullable<i32> loc(#loc14)
        %16 = db.cast %15 : !db.nullable<i32> -> !db.nullable<!db.decimal<19, 0>> loc(#loc15)
        %17 = db.mul %14 : !db.nullable<!db.decimal<18, 2>>, %16 : !db.nullable<!db.decimal<19, 0>> loc(#loc16)
        tuples.return %17 : !db.nullable<!db.decimal<38, 2>> loc(#loc12)
      } attributes {rows = 5860.5615234375 : f64} loc(#loc12)
      %12 = relalg.aggregation %11 [] computes : [@aggr0::@tmp_attr0({type = !db.nullable<!db.decimal<38, 2>>})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
        %14 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.nullable<!db.decimal<38, 2>> loc(#loc18)
        tuples.return %14 : !db.nullable<!db.decimal<38, 2>> loc(#loc19)
      } attributes {rows = 1.000000e+00 : f64} loc(#loc17)
      %13 = relalg.materialize %12 [@aggr0::@tmp_attr0] => ["revenue"] : !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc20)
      relalg.query_return %13 : !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc21)
    } -> !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc2)
    subop.set_result 0 %0 : !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc22)
    return loc(#loc23)
  } loc(#loc1)
} loc(#loc)
#loc = loc("input.mlir":1:0)
#loc1 = loc("input.mlir":2:2)
#loc2 = loc("input.mlir":3:4)
#loc3 = loc("input.mlir":4:6)
#loc4 = loc("input.mlir":5:6)
#loc5 = loc(unknown)
#loc6 = loc("input.mlir":6:8)
#loc7 = loc("input.mlir":8:8)
#loc8 = loc("input.mlir":12:8)
#loc9 = loc("input.mlir":14:8)
#loc10 = loc("input.mlir":18:8)
#loc11 = loc("input.mlir":20:8)
#loc12 = loc("input.mlir":27:6)
#loc13 = loc("input.mlir":28:8)
#loc14 = loc("input.mlir":29:8)
#loc15 = loc("input.mlir":30:8)
#loc16 = loc("input.mlir":31:8)
#loc17 = loc("input.mlir":34:6)
#loc18 = loc("input.mlir":35:8)
#loc19 = loc("input.mlir":36:8)
#loc20 = loc("input.mlir":38:6)
#loc21 = loc("input.mlir":39:6)
#loc22 = loc("input.mlir":41:4)
#loc23 = loc("input.mlir":42:4)