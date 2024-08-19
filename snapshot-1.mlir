module attributes {subop.sequential} {
  func.func @main() {
    %0 = subop.execution_group (){
      %1 = subop.get_external "{ \22table\22: \22lineorder\22, \22mapping\22: { \22lo_commitdate$0\22 :\22lo_commitdate\22,\22lo_custkey$0\22 :\22lo_custkey\22,\22lo_discount$0\22 :\22lo_discount\22,\22lo_extendedprice$0\22 :\22lo_extendedprice\22,\22lo_linenumber$0\22 :\22lo_linenumber\22,\22lo_orderdate$0\22 :\22lo_orderdate\22,\22lo_orderkey$0\22 :\22lo_orderkey\22,\22lo_orderpriority$0\22 :\22lo_orderpriority\22,\22lo_ordtotalprice$0\22 :\22lo_ordtotalprice\22,\22lo_partkey$0\22 :\22lo_partkey\22,\22lo_quantity$0\22 :\22lo_quantity\22,\22lo_revenue$0\22 :\22lo_revenue\22,\22lo_shipmode$0\22 :\22lo_shipmode\22,\22lo_shippriority$0\22 :\22lo_shippriority\22,\22lo_suppkey$0\22 :\22lo_suppkey\22,\22lo_supplycost$0\22 :\22lo_supplycost\22,\22lo_tax$0\22 :\22lo_tax\22} }" : !subop.table<[lo_commitdate$0 : !db.nullable<i32>, lo_custkey$0 : !db.nullable<i32>, lo_discount$0 : !db.nullable<i32>, lo_extendedprice$0 : !db.nullable<!db.decimal<18, 2>>, lo_linenumber$0 : !db.nullable<i32>, lo_orderdate$0 : !db.nullable<i32>, lo_orderkey$0 : !db.nullable<i32>, lo_orderpriority$0 : !db.nullable<!db.string>, lo_ordtotalprice$0 : !db.nullable<!db.decimal<18, 2>>, lo_partkey$0 : !db.nullable<i32>, lo_quantity$0 : !db.nullable<i32>, lo_revenue$0 : !db.nullable<!db.decimal<18, 2>>, lo_shipmode$0 : !db.nullable<!db.string>, lo_shippriority$0 : !db.nullable<!db.char<1>>, lo_suppkey$0 : !db.nullable<i32>, lo_supplycost$0 : !db.nullable<!db.decimal<18, 2>>, lo_tax$0 : !db.nullable<i32>]> loc(#loc3)
      %2 = subop.scan %1 : !subop.table<[lo_commitdate$0 : !db.nullable<i32>, lo_custkey$0 : !db.nullable<i32>, lo_discount$0 : !db.nullable<i32>, lo_extendedprice$0 : !db.nullable<!db.decimal<18, 2>>, lo_linenumber$0 : !db.nullable<i32>, lo_orderdate$0 : !db.nullable<i32>, lo_orderkey$0 : !db.nullable<i32>, lo_orderpriority$0 : !db.nullable<!db.string>, lo_ordtotalprice$0 : !db.nullable<!db.decimal<18, 2>>, lo_partkey$0 : !db.nullable<i32>, lo_quantity$0 : !db.nullable<i32>, lo_revenue$0 : !db.nullable<!db.decimal<18, 2>>, lo_shipmode$0 : !db.nullable<!db.string>, lo_shippriority$0 : !db.nullable<!db.char<1>>, lo_suppkey$0 : !db.nullable<i32>, lo_supplycost$0 : !db.nullable<!db.decimal<18, 2>>, lo_tax$0 : !db.nullable<i32>]> {lo_discount$0 => @lineorder::@lo_discount({type = !db.nullable<i32>}), lo_extendedprice$0 => @lineorder::@lo_extendedprice({type = !db.nullable<!db.decimal<18, 2>>}), lo_orderdate$0 => @lineorder::@lo_orderdate({type = !db.nullable<i32>}), lo_quantity$0 => @lineorder::@lo_quantity({type = !db.nullable<i32>})} loc(#loc3)
      %3 = subop.map %2 computes : [@map::@pred({type = i1})] input : [@lineorder::@lo_orderdate] (%arg0: !db.nullable<i32>){
        %27 = db.constant(19940204 : i32) : i32 loc(#loc5)
        %28 = db.constant(19940210 : i32) : i32 loc(#loc6)
        %29 = db.between %arg0 : !db.nullable<i32> between %27 : i32, %28 : i32, lowerInclusive : true, upperInclusive : true loc(#loc7)
        %30 = db.derive_truth %29 : !db.nullable<i1> loc(#loc4)
        tuples.return %30 : i1 loc(#loc4)
      } loc(#loc4)
      %4 = subop.filter %3 all_true [@map::@pred] {selectivity = 9.765625E-4 : f64} loc(#loc4)
      %5 = subop.map %4 computes : [@map_u_1::@pred({type = i1})] input : [@lineorder::@lo_discount] (%arg0: !db.nullable<i32>){
        %27 = db.constant(5 : i32) : i32 loc(#loc9)
        %28 = db.constant(7 : i32) : i32 loc(#loc10)
        %29 = db.between %arg0 : !db.nullable<i32> between %27 : i32, %28 : i32, lowerInclusive : true, upperInclusive : true loc(#loc11)
        %30 = db.derive_truth %29 : !db.nullable<i1> loc(#loc8)
        tuples.return %30 : i1 loc(#loc8)
      } loc(#loc8)
      %6 = subop.filter %5 all_true [@map_u_1::@pred] {selectivity = 0.0947265625 : f64} loc(#loc8)
      %7 = subop.map %6 computes : [@map_u_2::@pred({type = i1})] input : [@lineorder::@lo_quantity] (%arg0: !db.nullable<i32>){
        %27 = db.constant(26 : i32) : i32 loc(#loc13)
        %28 = db.constant(35 : i32) : i32 loc(#loc14)
        %29 = db.between %arg0 : !db.nullable<i32> between %27 : i32, %28 : i32, lowerInclusive : true, upperInclusive : true loc(#loc15)
        %30 = db.derive_truth %29 : !db.nullable<i1> loc(#loc12)
        tuples.return %30 : i1 loc(#loc12)
      } loc(#loc12)
      %8 = subop.filter %7 all_true [@map_u_2::@pred] {selectivity = 0.1572265625 : f64} loc(#loc12)
      %9 = subop.map %8 computes : [@map_u_3::@pred({type = i1})] input : [@lineorder::@lo_orderdate] (%arg0: !db.nullable<i32>){
        %27 = db.isnull %arg0 : <i32> loc(#loc17)
        %28 = db.not %27 : i1 loc(#loc18)
        tuples.return %28 : i1 loc(#loc16)
      } loc(#loc16)
      %10 = subop.filter %9 all_true [@map_u_3::@pred] {selectivity = 1.000000e+00 : f64} loc(#loc16)
      %11 = subop.map %10 computes : [@map_u_4::@pred({type = i1})] input : [@lineorder::@lo_discount] (%arg0: !db.nullable<i32>){
        %27 = db.isnull %arg0 : <i32> loc(#loc20)
        %28 = db.not %27 : i1 loc(#loc21)
        tuples.return %28 : i1 loc(#loc19)
      } loc(#loc19)
      %12 = subop.filter %11 all_true [@map_u_4::@pred] {selectivity = 1.000000e+00 : f64} loc(#loc19)
      %13 = subop.map %12 computes : [@map_u_5::@pred({type = i1})] input : [@lineorder::@lo_quantity] (%arg0: !db.nullable<i32>){
        %27 = db.isnull %arg0 : <i32> loc(#loc23)
        %28 = db.not %27 : i1 loc(#loc24)
        tuples.return %28 : i1 loc(#loc22)
      } loc(#loc22)
      %14 = subop.filter %13 all_true [@map_u_5::@pred] {selectivity = 1.000000e+00 : f64} loc(#loc22)
      %15 = subop.map %14 computes : [@map_u_6::@pred({type = i1})] input : [@lineorder::@lo_discount] (%arg0: !db.nullable<i32>){
        %27 = db.isnull %arg0 : <i32> loc(#loc26)
        %28 = db.not %27 : i1 loc(#loc27)
        tuples.return %28 : i1 loc(#loc25)
      } loc(#loc25)
      %16 = subop.filter %15 all_true [@map_u_6::@pred] {selectivity = 1.000000e+00 : f64} loc(#loc25)
      %17 = subop.map %16 computes : [@map_u_7::@pred({type = i1})] input : [@lineorder::@lo_orderdate] (%arg0: !db.nullable<i32>){
        %27 = db.isnull %arg0 : <i32> loc(#loc29)
        %28 = db.not %27 : i1 loc(#loc30)
        tuples.return %28 : i1 loc(#loc28)
      } loc(#loc28)
      %18 = subop.filter %17 all_true [@map_u_7::@pred] {selectivity = 1.000000e+00 : f64} loc(#loc28)
      %19 = subop.map %18 computes : [@map_u_8::@pred({type = i1})] input : [@lineorder::@lo_quantity] (%arg0: !db.nullable<i32>){
        %27 = db.isnull %arg0 : <i32> loc(#loc32)
        %28 = db.not %27 : i1 loc(#loc33)
        tuples.return %28 : i1 loc(#loc31)
      } loc(#loc31)
      %20 = subop.filter %19 all_true [@map_u_8::@pred] {selectivity = 1.000000e+00 : f64} loc(#loc31)
      %21 = subop.map %20 computes : [@map0::@tmp_attr1({type = !db.nullable<!db.decimal<38, 2>>})] input : [@lineorder::@lo_extendedprice,@lineorder::@lo_discount] (%arg0: !db.nullable<!db.decimal<18, 2>>,%arg1: !db.nullable<i32>){
        %27 = db.cast %arg1 : !db.nullable<i32> -> !db.nullable<!db.decimal<19, 0>> loc(#loc35)
        %28 = db.mul %arg0 : !db.nullable<!db.decimal<18, 2>>, %27 : !db.nullable<!db.decimal<19, 0>> loc(#loc36)
        tuples.return %28 : !db.nullable<!db.decimal<38, 2>> loc(#loc37)
      } loc(#loc34)
      %22 = subop.create_simple_state <[aggrval$0 : !db.nullable<!db.decimal<38, 2>>]> initial : {
        %27 = db.null : <!db.decimal<38, 2>> loc(#loc38)
        tuples.return %27 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
      } loc(#loc38)
      %23 = subop.lookup %21%22 [] : !subop.simple_state<[aggrval$0 : !db.nullable<!db.decimal<38, 2>>]> @lookup::@ref({type = !subop.lookup_entry_ref<!subop.simple_state<[aggrval$0 : !db.nullable<!db.decimal<38, 2>>]>>}) loc(#loc38)
      subop.reduce %23 @lookup::@ref[@map0::@tmp_attr1] ["aggrval$0"] ([%arg0],[%arg1]){
        %27 = db.isnull %arg1 : <!db.decimal<38, 2>> loc(#loc38)
        %28 = db.isnull %arg0 : <!db.decimal<38, 2>> loc(#loc38)
        %29 = db.add %arg1 : !db.nullable<!db.decimal<38, 2>>, %arg0 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
        %30 = arith.select %28, %arg1, %29 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
        %31 = arith.select %27, %arg0, %30 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
        tuples.return %31 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
      }combine: ([%arg0],[%arg1]){
        %27 = db.isnull %arg0 : <!db.decimal<38, 2>> loc(#loc38)
        %28 = db.isnull %arg1 : <!db.decimal<38, 2>> loc(#loc38)
        %29 = db.constant(0 : i64) : !db.decimal<38, 2> loc(#loc38)
        %30 = db.as_nullable %29 : !db.decimal<38, 2> -> <!db.decimal<38, 2>> loc(#loc38)
        %31 = arith.select %27, %30, %arg0 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
        %32 = arith.select %28, %30, %arg1 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
        %33 = db.add %31 : !db.nullable<!db.decimal<38, 2>>, %32 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
        %34 = arith.andi %27, %28 : i1 loc(#loc38)
        %35 = arith.select %34, %arg0, %33 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
        tuples.return %35 : !db.nullable<!db.decimal<38, 2>> loc(#loc38)
      } loc(#loc38)
      %24 = subop.scan %22 : !subop.simple_state<[aggrval$0 : !db.nullable<!db.decimal<38, 2>>]> {aggrval$0 => @aggr0::@tmp_attr0({type = !db.nullable<!db.decimal<38, 2>>})} loc(#loc38)
      %25 = subop.create !subop.result_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>]> loc(#loc39)
      subop.materialize %24 {@aggr0::@tmp_attr0 => revenue$0}, %25 : !subop.result_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>]> loc(#loc39)
      %26 = subop.create_from ["revenue"] %25 : !subop.result_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>]> -> !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc39)
      subop.execution_group_return %26 : !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc40)
    } -> !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc2)
    subop.set_result 0 %0 : !subop.local_table<[revenue$0 : !db.nullable<!db.decimal<38, 2>>], ["revenue"]> loc(#loc41)
    return loc(#loc42)
  } loc(#loc1)
} loc(#loc)
#loc = loc("snapshot-0.mlir":1:0)
#loc1 = loc("snapshot-0.mlir":2:2)
#loc2 = loc("snapshot-0.mlir":3:4)
#loc3 = loc("snapshot-0.mlir":4:6)
#loc4 = loc("snapshot-0.mlir":5:6)
#loc5 = loc("snapshot-0.mlir":7:8)
#loc6 = loc("snapshot-0.mlir":6:8)
#loc7 = loc("snapshot-0.mlir":9:8)
#loc8 = loc("snapshot-0.mlir":12:6)
#loc9 = loc("snapshot-0.mlir":14:8)
#loc10 = loc("snapshot-0.mlir":13:8)
#loc11 = loc("snapshot-0.mlir":16:8)
#loc12 = loc("snapshot-0.mlir":19:6)
#loc13 = loc("snapshot-0.mlir":21:8)
#loc14 = loc("snapshot-0.mlir":20:8)
#loc15 = loc("snapshot-0.mlir":23:8)
#loc16 = loc("snapshot-0.mlir":26:6)
#loc17 = loc("snapshot-0.mlir":28:8)
#loc18 = loc("snapshot-0.mlir":29:8)
#loc19 = loc("snapshot-0.mlir":32:6)
#loc20 = loc("snapshot-0.mlir":34:8)
#loc21 = loc("snapshot-0.mlir":35:8)
#loc22 = loc("snapshot-0.mlir":38:6)
#loc23 = loc("snapshot-0.mlir":40:8)
#loc24 = loc("snapshot-0.mlir":41:8)
#loc25 = loc("snapshot-0.mlir":44:6)
#loc26 = loc("snapshot-0.mlir":46:8)
#loc27 = loc("snapshot-0.mlir":47:8)
#loc28 = loc("snapshot-0.mlir":50:6)
#loc29 = loc("snapshot-0.mlir":52:8)
#loc30 = loc("snapshot-0.mlir":53:8)
#loc31 = loc("snapshot-0.mlir":56:6)
#loc32 = loc("snapshot-0.mlir":58:8)
#loc33 = loc("snapshot-0.mlir":59:8)
#loc34 = loc("snapshot-0.mlir":62:6)
#loc35 = loc("snapshot-0.mlir":65:8)
#loc36 = loc("snapshot-0.mlir":66:8)
#loc37 = loc("snapshot-0.mlir":67:8)
#loc38 = loc("snapshot-0.mlir":69:6)
#loc39 = loc("snapshot-0.mlir":73:6)
#loc40 = loc("snapshot-0.mlir":74:6)
#loc41 = loc("snapshot-0.mlir":76:4)
#loc42 = loc("snapshot-0.mlir":77:4)
