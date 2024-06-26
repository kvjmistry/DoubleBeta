# MC
basename="../data/NEXT100_Ar_alpha_nexus_5.6MeV"
city buffy ../config/buffy.conf -i ${basename}.h5 -o ${basename}_buffy.h5
city hypathia ../config/hypathia.conf -i ${basename}_buffy.h5 -o ${basename}_hypathia.h5

basename="../data/NEXT100_Ar_alpha_profile_5.6MeV"
city buffy ../config/buffy.conf -i ${basename}.h5 -o ${basename}_buffy.h5
city hypathia ../config/hypathia.conf -i ${basename}_buffy.h5 -o ${basename}_hypathia.h5

# Data
basename="../data/run_13837_0000_ldc1_trg0"
city irene ../config/irene.conf -i ${basename}.waveforms.h5 -o ${basename}_irene.h5
city sophronia ../config/sophronia.conf -i ${basename}_irene.h5 -o ${basename}_sophronia.h5