 no_attack= [36.45,42.12,40.56,40.32,40.30];
 pgd1=[31.66,40.70,39.55,38.64,37.72];
 pgd5=[30.03,38.10,36.19,35.57,34.86];
 ST_no_attack = [36.57,36.57,36.57,36.57,36.57];
 ST_pgd1 = [29.97,29.97,29.97,29.97,29.97];
 ST_pgd5= [26.52,26.52,26.52,26.52,26.52];
 x=[0,1,2,5,10];
 marksize = 7;
 plot(x,[no_attack],'-o','LineWidth',2,'MarkerSize',marksize)
 hold on;
 plot(x,[ST_no_attack],'-v','LineWidth',2,'MarkerSize',marksize)
 xticks([0,1,2,5,10]);
 xticklabels({'AT','1','2','5','10'});
 legend('Advcode training','ST');
 xlabel({'Advcode generation','(From high frequency to low frequency)'})
 ylabel({'F1 without attack','(accuracy evaluation)'})
 print(gcf,'-dpng','epoch_vs_accuracy.png');
 hold off;
 
 plot(x,[pgd1],'-o','LineWidth',2,'MarkerSize',marksize);
 hold on;
 plot(x,[ST_pgd1],'-v','LineWidth',2,'MarkerSize',marksize);
 xticks([0 ,1,2,5,10]);
 xticklabels({'AT','1','2','5','10'});
 legend('Advcode training','ST');
 xlabel({'Advcode generation','(From high frequency to low frequency)'})
 ylabel({'F1 with UWisc-pgd-3 attack (1 site)','(robustness evaluation)'})
 print(gcf,'-dpng','epoch_vs_robustness_1.png');
 hold off;
 plot(x,[pgd5],'-o','LineWidth',2,'MarkerSize',marksize);
 hold on;
 plot(x,[ST_pgd5],'-v','LineWidth',2,'MarkerSize',marksize);
 xticks([0 ,1,2,5,10]);
 xticklabels({'AT','1','2','5','10'});
 legend('Advcode training','ST');
 xlabel({'Advcode generation','(From high frequency to low frequency)'})
 ylabel({'F1 with UWisc-pgd-3 attack (5 sites)','(robustness evaluation)'})
 print(gcf,'-dpng','epoch_vs_robustness_5.png');