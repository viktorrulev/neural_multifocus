function [dataset_blurred, dataset_sharp] = dataset_from_photo(photo_rgb, labeled_photo, step)

    area_r = 64;
    max_shift = ceil(step / 2);

    photo = rgb2gray(photo_rgb);
    
    photo_height = size(photo, 1);
    photo_width = size(photo, 2);
    
    height_lim = [1 + area_r + max_shift, photo_height - area_r - max_shift];
    width_lim  = [1 + area_r + max_shift, photo_width  - area_r - max_shift];
    
    counter = 0;
    counter_sharp = 0;
    counter_blurred = 0;
    
    for i = height_lim(1):step:height_lim(2)
        for j = width_lim(1):step:width_lim(2)
            shift_i = round((rand() - 0.5) * 2 * max_shift);
            shift_j = round((rand() - 0.5) * 2 * max_shift);
            
            ti = i + shift_i;
            tj = j + shift_j;
            
            px = labeled_photo(ti, tj, 1:3);
            px = reshape(px, [1 3]);
            
            if px(1) - px(2) > 50 || px(2) - px(1) > 50
                if mod(counter, 1000) == 0
                    fprintf('%d: %d blurred, %d sharp\n', counter, counter_blurred, counter_sharp);
                end;
                
                photo_fragment = photo(ti - area_r:ti + area_r, tj - area_r:tj + area_r);
                small_photo = imresize(photo_fragment, [64 64]);
                [photo_grad, ~] = imgradient(small_photo);

                if px(1) - px(2) > 50
                    counter = counter + 1;
                    if mod(counter, 3) == 0
                        counter_blurred = counter_blurred + 1;
                        dataset_blurred{counter_blurred}.photo = photo_grad;
                        dataset_blurred{counter_blurred}.index = [ti, tj];
                        dataset_blurred{counter_blurred}.class = 0;
                    end;
                end;

                if px(2) - px(1) > 50
                    counter = counter + 1;
                    counter_sharp = counter_sharp + 1;
                    dataset_sharp{counter_sharp}.photo = photo_grad;
                    dataset_sharp{counter_sharp}.index = [ti, tj];
                    dataset_sharp{counter_sharp}.class = 1;
                end;
            end;
        end;
    end;
    
    fprintf('%d examples extracted (%d blurred, %d sharp)\n', counter, counter_blurred, counter_sharp);
    
    
    
    