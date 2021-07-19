save_path = "./nc_20210224_process/"

options(BioC_mirror="http://mirrors.tuna.tsinghua.edu.cn/bioconductor/")
options("repos"=c(CRAN="http://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# change the "FALSE" to "TURE" to install R package
if(FALSE)
{
  # GEOquery
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("GEOquery")

  # AnnoProbe
  install.packages('devtools')
  library(devtools)
  install_github('jmzeng1314/AnnoProbe')
}

library(GEOquery)
library(AnnoProbe)

description_to_label <- function(des){
  output_list = list()
  for (i in 1:dim(des)[1]){
    des_each = des[i, 1]
    if (substring(des_each, 1, 19) == "bacterial pneumonia"){key = 1}
    else if (substring(des_each, 1, 16) == "Severe Influenza"){key = 2}
    else if (substring(des_each, 1, 3) == "Vac"){key = 0}
    
    else if (substring(des_each, 1, 5) == "Pande"){key = 2}
    else if (substring(des_each, 1, 5) == "Healt"){key = 0}
    
    else if (substring(des_each, 1, 9) == "blood_day"){key = 2}
    else if (substring(des_each, 1, 13) == "Blood_Control"){key = 0}
    
    
    else if (substring(des_each, 1, 24) == "Experiment_Post_Surgical") {key = 1}
    else if (substring(des_each, 1, 17) == "Experiment_Sepsis") {key = 1}
    else if (substring(des_each, 1, 15) == "Control_Healthy") {key = 0}
    
    else if (substring(des_each, 1, 25) == "PAXgene whole blood, SIRS"){key = 2}
    else if (substring(des_each, 1, 40) == "PAXgene whole blood, bacterial pneumonia"){key = 1}
    else if (substring(des_each, 1, 42) == "PAXgene whole blood, influenza A pneumonia"){key = 2}
    else if (substring(des_each, 1, 36) == "PAXgene whole blood, healthy control"){key = 0}
    else if (substring(des_each, 1, 62) == "PAXgene whole blood, mixed bacterial and influenza A pneumonia"){key = 3}

    
    else if (substring(des_each, 1, 7) == "Blood_P"){key = 1}
    else if (substring(des_each, 1, 8) == "Blood_HV"){key = 0}
    
    else if (substring(des_each, 1, 20) == "whole blood-BACTERIA"){key = 1}
    else if (substring(des_each, 1, 23) == "whole blood-COINFECTION"){key = 3}
    else if (substring(des_each, 1, 17) == "whole blood-VIRUS"){key = 2}
    else if (substring(des_each, 1, 27) == "whole blood-Healthy Control"){key = 0}
    
    else if (substring(des_each, 1, 15) == "healthy subject"){key = 0}
    else if (substring(des_each, 1, 27) == "intensive-care unit patient"){key = 1}
    
    # 69528
    else if (substring(des_each, 1, 18) == "Uninfected healthy") {key = 0}
    else if (substring(des_each, 1, 35) == "Uninfected type 2 diabetes mellitus") {key = 0}
    else if (substring(des_each, 1, 22) == "Septicemic melioidosis") {key = 1}
    else if (substring(des_each, 1, 12) == "Other sepsis") {key = 1}
    
    else if (des_each == "HC"){key = 0}
    else if (des_each == "H1N1"){key = 2}
    else if (des_each == "A"){key = 2}
    else if (des_each == "B"){key = 2}
    else if (des_each == "H3N2"){key = 2}
    
    else if (substring(des_each, 1, 19) == "PBMC_InfluenzaA_INF") {key=2}
    else if (substring(des_each, 1, 19) == "PBMC_InfluenzaB_INF") {key=2}
    else if (substring(des_each, 1, 22) == "PBMC_S.aureus_MRSA_INF") {key=1}
    else if (substring(des_each, 1, 21) == "PBMC_S.pneumoniae_INF") {key=1}
    else if (substring(des_each, 1, 22) == "PBMC_S.aureus_MSSA_INF") {key=1}
    
    
    else if (substring(des_each, 1, 23) == "bacterial pneumonia_day") {key=1}
    else if (substring(des_each, 1, 20) == "Severe Influenza_day") {key=2}
    else if (substring(des_each, 1, 3) == "Vac") {key=0} 
    
    else if (substring(des_each, 1, 20) == "pathogen: Adenovirus") {key=2}
    else if (substring(des_each, 1, 14) == "pathogen: HHV6") {key=2}
    else if (substring(des_each, 1, 21) == "pathogen: Enterovirus") {key=2}
    else if (substring(des_each, 1, 20) == "pathogen: Rhinovirus") {key=2}
    else if (substring(des_each, 1, 16) == "pathogen: E.coli") {key=1}
    else if (substring(des_each, 1, 18) == "pathogen: Bacteria") {key=1}
    else if (substring(des_each, 1, 14) == "pathogen: MRSA") {key=1}
    else if (substring(des_each, 1, 20) == "pathogen: Salmonella") {key=1}
    else if (substring(des_each, 1, 14) == "pathogen: MSSA") {key=1}
    else if (substring(des_each, 1, 14) == "pathogen: None") {key=0}
    
    else if (substring(des_each, 1, 7) == "WB-bact") {key=1}
    else if (substring(des_each, 1, 10) == "WB-control") {key=0}
    else if (substring(des_each, 1, 7) == "WB-H1N1") {key=2}
    else if (substring(des_each, 1, 6) == "WB-RSV") {key=2}
    
    # 不一定可以使用 
    else if (substring(des_each, 1, 16) == "disease: Control") {key=0}
    else if (substring(des_each, 1, 20) == "disease: SepticShock") {key=1}
    else if (substring(des_each, 1, 13) == "disease: SIRS") {key=2}
    else if (substring(des_each, 1, 15) == "disease: Sepsis") {key=1}
    
    
    else if (substring(des_each, 1, 40) == "non-infectious illness") {key=0}
    else if (substring(des_each, 1, 27) == "bacterial") {key=1}
    else if (substring(des_each, 1, 23) == "viral") {key=2}
    
    # 25504
    else if (substring(des_each, 1, 3) == "Con") {key = 0}
    else if (substring(des_each, 1, 3) == "Inf") {key = 1}
    else if (substring(des_each, 1, 3) == "NEC") {key = 1}
    else if (substring(des_each, 1, 3) == "Vir") {key = 2}
    
    # 42834
    else if (substring(des_each, 1, 2) == "TB") {key=1}  # Tuberculosis
    else if (substring(des_each, 1, 7) == "Sarcoid") {key=10}
    else if (substring(des_each, 1, 9) == "Pneumonia") {key=10}
    else if (substring(des_each, 1, 6) == "Cancer") {key=10}
    
    else if (substring(des_each, 1, 61) == "infection: our tests did not detect one of the viruses sought") {key=10}
    else if (substring(des_each, 1, 40) == "infection: respiratory syncytial virus A") {key=2}  
    else if (substring(des_each, 1, 22) == "infection: enterovirus") {key=2}  
    else if (substring(des_each, 1, 27) == "infection: human rhinovirus") {key=2}  
    else if (substring(des_each, 1, 33) == "infection: human coronavirus HKU1") {key=2}  
    else if (substring(des_each, 1, 33) == "infection: human coronavirus NL63") {key=2}  
    else if (substring(des_each, 1, 28) == "infection: influenza B virus") {key=2}  
    else if (substring(des_each, 1, 28) == "infection: influenza A virus") {key=2} 

    else {
      print("Unseen characters appeared");
      print(des_each);
      readline()
    }
    output_list[i] <- key
  }
  return (unlist(output_list))
}


get_detail_func <- function(GSE_ID_each){
  print(GSE_ID_each)
  GSE_ID_each_cated = paste("GSE", GSE_ID_each, sep="")
  print(GSE_ID_each_cated)
  # step1 Probe expression matrix for acquiring data
  gset=AnnoProbe::geoChina(GSE_ID_each_cated)
  # suppressWarnings(load(paste("./", GSE_ID_each_cated,"_eSet.Rdata", sep="")))  # Apply the gset in the current path
  eSet=gset[[1]]
  probes_expr_without_log2 <- exprs(eSet);dim(probes_expr_without_log2)
  probes_expr_with_log2=log2(probes_expr_without_log2+1)
  phenoDat <- pData(eSet)
  write.table(phenoDat, file=paste(save_path, "all_lc_", GSE_ID_each_cated, ".csv",sep=""), sep=",", row.names=T, quote=FALSE)
  # step2 Annotate the probe in the platform file from which the data was obtained
  gpl=eSet@annotation
  checkGPL(gpl)
  printGPLInfo(gpl)
  probe2gene=idmap(gpl)
  head(probe2gene)
  genes_expr_with_log2 <-filterEM(probes_expr_with_log2,probe2gene)
  genes_expr_without_log2 <-filterEM(probes_expr_without_log2,probe2gene)
  
  if (GSE_ID_each == "69528"){
    phenoDat_col1 = phenoDat["study group:ch1"]
    print("phenoDat_col1   69528")
    print(phenoDat_col1)
    phenoLabel<-within(phenoDat_col1,{
      Label<-description_to_label(phenoDat_col1["study group:ch1"])
    })
  }
  else if (GSE_ID_each == "111368"){
    phenoDat_col1 = phenoDat["flu_type:ch1"]
    print("phenoDat_col1   111368")
    print(phenoDat_col1)
    phenoLabel<-within(phenoDat_col1,{
      Label<-description_to_label(phenoDat_col1["flu_type:ch1"])
    })
  }  
  else if (GSE_ID_each == "40396"){
    phenoDat_col1 = phenoDat["characteristics_ch1.4"]
    print("phenoDat_col1   40396")
    print(phenoDat_col1)
    phenoLabel<-within(phenoDat_col1,{
      Label<-description_to_label(phenoDat_col1["characteristics_ch1.4"])
    })
  }
  else if (GSE_ID_each == "66099"){
    phenoDat_col1 = phenoDat["characteristics_ch1.2"]
    print("phenoDat_col1   66099")
    print(phenoDat_col1)
    phenoLabel<-within(phenoDat_col1,{
      Label<-description_to_label(phenoDat_col1["characteristics_ch1.2"])
    })
  }
  else if (GSE_ID_each == "63990"){
    phenoDat_col1 = phenoDat["infection_status:ch1"]
    print("phenoDat_col1   63990")
    print(phenoDat_col1)
    phenoLabel<-within(phenoDat_col1,{
      Label<-description_to_label(phenoDat_col1["infection_status:ch1"])
    })
  }
  else if (GSE_ID_each == "68310"){
    phenoDat_col1 = phenoDat["characteristics_ch1.3"]
    print("phenoDat_col1   68310")
    print(phenoDat_col1)
    phenoLabel<-within(phenoDat_col1,{
      Label<-description_to_label(phenoDat_col1["characteristics_ch1.3"])
    })
  }
  else{
    phenoDat_col1 = phenoDat[1]
    print("phenoDat_col1")
    print(phenoDat_col1)
    phenoLabel<-within(phenoDat_col1,{
      Label<-description_to_label(phenoDat_col1["title"])
    })
  }
  write.table(genes_expr_without_log2,file=paste(save_path, "exp_gene_", GSE_ID_each_cated ,".txt", sep=""),sep="\t",row.name=T,quote=FALSE)
  write.table(genes_expr_with_log2, file=paste(save_path, "exp_gene_", GSE_ID_each_cated, "_log2.txt",sep=""), sep="\t", row.names=T, quote=FALSE)
  write.table(phenoLabel, file=paste(save_path, "label_", GSE_ID_each_cated, ".txt",sep=""), sep="\t", row.names=T, quote=FALSE)
}

# GSE_IDs = c("20346", "40012", "40396", "42026", "60244", "66099", "63990")  # coconut
GSE_IDs = c("21802", "27131", "28750", "42834", "57065", "68310", "69528", "111368")  # nc 2020
for (GSE_ID_each in GSE_IDs){
  print(GSE_ID_each)
  get_detail_func(GSE_ID_each)
}

