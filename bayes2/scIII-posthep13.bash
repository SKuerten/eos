#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=800
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"
export PMC_ADJUST_SAMPLE_SIZE=1
export PMC_CHUNKSIZE=4000
export PMC_CLUSTERS=50
export PMC_FINAL_CHUNKSIZE=250000
export PMC_GROUP_BY_RVALUE=2.35
export PMC_PATCH_LENGTH=200
export PMC_SKIP_INITIAL=0.5
# 0=D,1=B,2=A,3=C
export PMC_IGNORE_GROUPS="
    --pmc-ignore-group 3
    --pmc-ignore-group 1
    --pmc-ignore-group 2
"

export PMC_NUMBER_OF_JOBS=200
export LL_QUEUE=serial
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=60
# export PMC_CLIENT_ARGV="--resume-samples --step 3"

export PMC_UNCERTAINTY_INPUT="${BASE_NAME}/scIII_posthep13/pmc_parameter_samples_15.hdf5_merge"

# initial values
export GOF_MODE_0="{ +0.2139980138676844 +0.1067330691664306 -0.9015773142896008 +0.4204228261424409 -3.5489318231407530 -4.3416255139700075 +0.8153487123193911 +0.2250698763291053 +0.1267347991283046 +0.3790157248734963 +1.2480628351447394 +4.1552544185019302 +0.2308609560770084 +0.3758667296832123 -5.0675046115714268 +0.9321181493870555 +0.7884486829974310 +0.2666279546316401 +0.2450757919077328 +0.2614891116816296 -1.1556398390496629 +0.0568512694899802 +0.0861760382401733 -0.1536882172254353 +1.1021754145091565 +0.9837351017332779 +0.9766139877031543 +1.0384869101780134 +0.2896142789742939 -3.2234819547452940 +0.0481469025161157 -0.0235895379595761 +0.3280386906047750 +0.3472225771513271 }"
export GOF_MODE_1="{ +0.4798082031595698 -4.2782284817936054 +4.4983780205117414 +0.1161617586112098 +0.3340860700066529 +0.8428088957870614 +0.8095380546482196 +0.2254471158499912 +0.1456580895507022 +0.4128421373403845 +1.2596027654434609 +4.1717176994998857 +0.2284809027883456 +0.4066880541639691 -4.4111791828692164 +0.7710372408991124 +0.8654216088898412 +0.2464884262054280 +0.0776288556375827 +0.2328231561099281 -0.9253723507785300 -0.1047140054014695 +0.1182379674717815 +0.0375418582917963 +0.9644066684590203 +0.9914324922517825 +0.9733940926898171 +0.9817772592197517 +0.2704621457799316 -3.6533537906174365 +0.0171013032741150 -0.0392682514229755 +0.5246345804122408 +0.3506592682222369 }"
export GOF_MODE_2="{ -0.3291709124203963 +3.6707549812990754 -4.8988563474247471 -0.0843548810633417 -1.4022806628600688 -0.6263805138540568 +0.7920437468939056 +0.2257062402118623 +0.0892876696318071 +0.4173400137988895 +1.2874900690241724 +4.2078484541481487 +0.2236005130729612 +0.3720858965529553 -4.6255855152852847 +0.8852741750911690 +1.0655736662994924 +0.2347856634322430 +0.8446688775285355 +0.2196960717669310 -1.6410362715112889 -0.0125970171797625 -0.1288930214643196 -0.0928743690756243 +0.8936273840038885 +0.9852237623095010 +0.9633896102504206 +0.8781435210912867 +0.2844409447315632 -3.2597545121488660 -0.0485124930155889 +0.3220689679388349 +0.4410650605245722 +0.3515560982038592 }"
export GOF_MODE_3="{ -0.0196206159229466 -1.3548361885323852 +0.5347804758779826 -0.4262677168225154 +4.2136558391006238 +4.4094851576091099 +0.7889799880126785 +0.2250059693820696 +0.0903445597527580 +0.4120641574680449 +1.2886019016103620 +4.1897564597761345 +0.2275725766013242 +0.4191263504379997 -4.6457532321000148 +0.9516981306829801 +0.7447208814416554 +0.2498264068850267 +0.8221034094396427 +0.2394633091850027 +0.2822775056976390 -0.2377868408122175 +0.0401063817605376 +0.0773179238338116 +0.9423593852448874 +0.8672588957296343 +0.7056997620173648 +0.9080911947299039 +0.3279823999316657 -2.7558664796790171 +0.0222806409589449 -0.6047528426587532 +0.5122067707791902 +0.3503130856215279 }"


export GOF_MODE_0="{
 +0.1508212580097578 +0.1897372098757005 -0.6023744149054687
 +0.4237873414046214 -3.9155364861377104 -4.4686368052186660
 +0.8108300168019551 +0.2251543251760519 +0.1307912432748354
 +0.3640547178856763 +1.2840134233364959 +4.1631277412712597
 +0.2289852554404269 +0.3829582487466998 -4.9513172918933499
 +0.9464962653989567 +0.8080758378370484 +0.2521235960711815
 +0.2850061029643475 +0.2426547096266008 -1.1266178645079381
 +0.0725252057827790 +0.0577302467099039 +0.0536921853491786
 +1.0438179389370967 +0.9568720846788398 +0.9682945294313665
 +1.0472628171503739 +0.2953089753224686 -3.2276320060568877
 +0.0156176280787133 +0.4700201824127542 +0.4599835578322292
 +0.3487125538521185 }"
export GOF_MODE_1="{
 +0.4891045191541478 -4.2679436027176978 +4.4476234333920459
 +0.1114889290368637 +0.5703468115735241 +0.8397671055132904
 +0.8091698484653452 +0.2254783140490567 +0.1446274365209099
 +0.3994901493815223 +1.2660142441188742 +4.1813803402743179
 +0.2285799365211049 +0.3969093124928803 -4.4514545134711065
 +0.7659057484899201 +0.8828762478582334 +0.2454425456642148
 +0.0002613861939412 +0.2344806610964877 -1.2600881602598888
 -0.0932064801602071 +0.1088023850742812 +0.0397931906245884
 +0.9483433569687749 +0.9776817171585557 +0.9721451291669085
 +0.9778616924776217 +0.2782378749171053 -3.5942843431763629
 +0.0157237397007821 +0.4568302255838959 +0.5329134040712227
 +0.3498874925482758 }"
export GOF_MODE_2="{
 -0.3294454156375440 +3.7033084792658708 -4.8352440600734292
 -0.0705098937666282 -1.0515469382939286 -0.5277211018937606
 +0.7946867002888391 +0.2257088766787508 +0.0909777042701995
 +0.4175538169512402 +1.2852696897213503 +4.2023790868482935
 +0.2247720308030648 +0.3698572151640381 -4.6476749137692019
 +0.8858662569438940 +1.0678200817270018 +0.2408789020998637
 +0.7064830208983474 +0.2297105563772162 -1.3539460523030098
 -0.0065607240331166 -0.1161988151211939 -0.0774373672905987
 +0.8949480487450954 +1.0009697695962170 +0.9687909686796629
 +0.9007931169772687 +0.2854144323839109 -3.4225029354151677
 -0.0271399663969550 -0.0815039809496151 +0.4499125806204070
 +0.3507321107940625 }"
export GOF_MODE_3="{
 -0.0157734421894522 -1.1321289138990118 +0.3796536153807689
 -0.4248361956296707 +4.2441573979695137 +4.4729213662276806
 +0.7978116193980115 +0.2249643129519471 +0.0926156925606630
 +0.4060723533857551 +1.2853420321140447 +4.1700261929234443
 +0.2290354980883414 +0.4111771664298736 -4.5845699901312740
 +0.9586165808536107 +0.7508914837753804 +0.2559300827761227
 +0.8858585212128008 +0.2509233005436464 +0.1712867098175950
 -0.0649753138726491 +0.0033480344787607 +0.0688262173108371
 +0.9440249842214682 +0.8472659785471416 +0.8536563581683839
 +1.1092438744765538 +0.3237493554639322 -2.7056328656334241
 -0.0211753548987642 -0.5981469881284946 +0.5061356452613027
 +0.3491244684576670 }"

main $@
