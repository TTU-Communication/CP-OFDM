hold on;

nColor = 3;
nStyle = 7;

colors = colororder;

colorTexts = {"p=0.001", "p=0.01", "p=0.1", "OFDM + IN + PGIR, T = 1.5", "OFDM + IN + PGIR, T = 2.0", "OFDM + IN + PGIR, T = 2.5"};
styleTexts = {"PGIR 20 time(s)", "Blanking", "Clipping", "Clipping + Blanking", "Deep Clipping", "Replacement", "Clipping + Replacement + Blanking", "K = 80", "K = 90", "K = 100", "K = 200"};
styleLines = {'-', '--', '-.', '-', '-', '-', '-', '-', '-', '-', '-'};
styleMarkers = {'none', 'none', 'none', 'o', '*', 'x', 'square', '^', 'pentagram', '|', 'diamond'};
% colorTexts = {"$\overline{\mathrm{NMSE}}$", "BER", "OFDM + IN + PGIR"};
% styleTexts = {"$|I_T|=14$", "$|I_T|=11$", "Have one DC $(|I_T|=15)$", "K = 50", "K = 60", "K = 70", "K = 80", "K = 90", "K = 100", "K = 200"};
% styleLines = {'-', '--', '-.', '-', '-', '-', '-', '-', '-', '-', '-'};
% styleMarkers = {'none', 'none', 'none', 'o', '*', 'x', 'square', '^', 'pentagram', '|', 'diamond'};

hColors = cell(nColor, 1);
hStyles = cell(nStyle, 1);

for idx = 1:nColor
    % hColors{idx} = bar(nan, nan, 1, FaceColor=colors(idx,:), ...
    %     EdgeColor="none", FaceAlpha=0.6, displayname=sprintf('|K_D^\\prime|=%d', idx));
    hColors{idx} = plot(nan, nan, LineStyle="-", Marker="none", LineWidth=1.5, ...
        Color=colors(idx,:), DisplayName=colorTexts{idx});
end

for idx = 1:nStyle
    hStyles{idx} = plot(nan, nan, Color="k", LineWidth=1.5, ...
        LineStyle=styleLines{idx}, Marker=styleMarkers{idx}, MarkerSize=8, ...
        displayname=styleTexts{idx});
end

legend([hColors{:}, hStyles{:}]);