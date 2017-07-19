gos_file = open('GOS.csv')
gos_lines = gos_file.readlines()
gos_file.close()

gos_map = {}
for l in gos_lines[1:]:
  sl = l.split(',')
  gos_map[int(sl[0])] = int(sl[1])

observe_file = open('rf_observ - Copy.csv')
obs_lines = observe_file.readlines()
observe_file.close()

obs_tot_map = {}
obs_rea_map = {}
obs_pre_map = {}
for l in obs_lines[1:]:
  sl = l.split(',')
  pid = int(sl[0][0:3])
  if pid not in obs_tot_map:
    obs_tot_map[pid] = 0
    obs_rea_map[pid] = 0
    obs_pre_map[pid] = 0
  obs_tot_map[pid] += 1
  obs_rea_map[pid] += int(sl[4])
  obs_pre_map[pid] += int(sl[5])

pen = open('outcome_study.csv', 'w')
pen.write('Patient,Total,Real,Pred')
for k, v in obs_tot_map.items():
  pen.write(str(k)+','+str(v)+','+str(obs_rea_map[k])+','+str(obs_pre_map[k])+'\n')
pen.close()