


void write_sightline_params(void)  {
	ofstream file;
	file.open(params_file, ios::out | ios::binary);
	
	for (int i = 0; i < Nhalo; i++)  {		
		for (int j = 0; j < Ndir; j++)  {
			for (int d = 0; d < 3; d++)  {
				file.write((char*)&start_inds[i][j][d], sizeof(int));
			}
			for (int d = 0; d < 2; d++)  {
				file.write((char*)&start_angles[i][j][d], sizeof(float));
			}
			file.write((char*)&halo_mass[i][j], sizeof(float));
		}
	}
	
}

}



void write_sightlines(void)  {
	ofstream file;
	file.open(sightline_file, ios::out | ios::binary);
	
	for (int i = 0; i < Nhalo; i++)  {		
		for (int j = 0; j < Ndir; j++)  {
			for (int s = 0; s < N_r; s++)  {
				file.write((char*)&rho_total_sl[i][j][s], sizeof(float));
				file.write((char*)&vel_sl[i][j][s],  sizeof(float));
			}
		}
	}
	
}



void write_spec_out(void)  {
	ofstream file, file2;
	file.open(wav_out_file, ios::out | ios::binary);
	file2.open(spec_out_file, ios::out | ios::binary);
	
	for (int w=0; w<N_wav; w++)  {
		file.write((char*)&wav_out[w],  sizeof(float));
	}
	
	for (int i=0; i < Nhalo; i++)  {
		for (int j=0; j < Ndir; j++)  {
			for (int w=0; w<N_wav; w++)  {
				for (int d=0; d<num_fields; d++) {
					file2.write((char*)&spec_out[i][j][w][d],  sizeof(float));
				}
			}
		}
	}
	file.close();
	
}



